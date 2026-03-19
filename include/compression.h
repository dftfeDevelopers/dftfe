/*
** compression.h - GPU Fixed-Rate Floating-Point Compressor
**
** Specialized for:
**   - 1D contiguous arrays only (stride = 1)
**   - Fixed-rate mode only (e.g. 8 or 12 bits/value)
**   - GPU-resident data (zero host<->device copies)
**   - User-specified device stream (non-blocking, async)
**   - float and double types
**   - Portable across CUDA, HIP, and SYCL backends
**
** Performance:
**   - For 8 and 12 bpv: 1 thread per block (bpt=1), no atomicAdd,
**     no memset. Stores as uint32_t (8bpv) or 3×uint16_t (12bpv).
**   - For other bpv: multi-block-per-thread super-block writes,
**     filling complete 64-bit Words for coalesced stores.
**
** Usage:
**   #include "compression.h"
**
**   size_t bytes = compression::compressed_size(N, 12);
**   // allocate d_comp with bytes on device, reuse
**
**   compression::compress(d_in, d_comp, N, 12, my_stream);
**   // ... NCCL/MPI send d_comp (bytes) ...
**   compression::decompress(d_comp, d_out, N, 12, my_stream);
**
** Bitstream layout compatible with zfp 1D fixed-rate (64-bit words).
**
** References:
* [1] P. Lindstrom, "Fixed-Rate Compressed Floating-Point Arrays," IEEE Trans.
*     Vis. Comput. Graph., vol. 20, no. 12, pp. 2674-2683, Dec. 2014.
*     DOI: 10.1109/TVCG.2014.2346458
*
* [2] P. Lindstrom, zfp: Compressed Floating-Point and Integer Arrays,
*     Lawrence Livermore National Laboratory, 2014-2024.
*     https://github.com/LLNL/zfp  (LLNL BSD-3-Clause)
*     The encode/decode algorithm, transforms, and bit-plane coding in this
*     file are derived from the zfp CUDA backend.
**
*/

#ifndef COMPRESSION_H
#define COMPRESSION_H

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

  /* =========================================================================
     Device: LocalBlockWriter -- write bits to a thread-local Word array

     Accumulates bits using |= (no atomics). The local buffer must be
     zero-initialized by the caller. This replaces the old AtomicBlockWriter
     that used atomicAdd to global memory.
     =========================================================================
   */

  struct LocalBlockWriter
  {
    unsigned int m_word_index;
    unsigned int m_start_bit;
    unsigned int m_current_bit;
    Word        *m_local_words;

    COMPRESSION_DEVICE_INLINE
    LocalBlockWriter(Word        *local_words,
                     unsigned int maxbits,
                     unsigned int local_block_idx)
      : m_current_bit(0)
      , m_local_words(local_words)
    {
      size_t bit_offset = (size_t)local_block_idx * maxbits;
      m_word_index      = (unsigned int)(bit_offset / WSIZE);
      m_start_bit       = (unsigned int)(bit_offset % WSIZE);
    }

    COMPRESSION_DEVICE_INLINE
    uint64
    write_bits(uint64 bits, unsigned int n_bits)
    {
      if (n_bits == 0)
        return bits;

      unsigned int seg_start = (m_start_bit + m_current_bit) % WSIZE;
      unsigned int write_index =
        m_word_index + (m_start_bit + m_current_bit) / WSIZE;
      unsigned int seg_end = seg_start + n_bits - 1;

      /* mask to lower n_bits - avoids UB shift-by-64 when n_bits == WSIZE */
      Word b = n_bits < WSIZE ? (bits & (((Word)1 << n_bits) - 1u)) : bits;
      m_local_words[write_index] |= (b << seg_start);

      if (seg_start < WSIZE && seg_end >= WSIZE)
        m_local_words[write_index + 1] |= (b >> (WSIZE - seg_start));

      m_current_bit += n_bits;
      return bits >> (Word)n_bits;
    }

    COMPRESSION_DEVICE_INLINE
    unsigned int
    write_bit(unsigned int bit)
    {
      unsigned int seg_start = (m_start_bit + m_current_bit) % WSIZE;
      unsigned int write_index =
        m_word_index + (m_start_bit + m_current_bit) / WSIZE;
      m_local_words[write_index] |= ((Word)bit << seg_start);
      m_current_bit += 1;
      return bit;
    }
  };

  /* =========================================================================
     Device: LocalBlockReader -- read bits from a thread-local Word array

     Used by the super-block decompress kernel: the caller loads wpt Words
     from global memory into a local array (L1-cached), then decodes all bpt
     blocks from that array using plain (non-__ldg) loads.  Carries the same
     advance-first invariant as BlockReader so decode_ints/decode_block work
     with either reader type via template dispatch.
     =========================================================================
   */

  struct LocalBlockReader
  {
    int         m_current_bit;
    const Word *m_words;
    Word        m_buffer;

    COMPRESSION_DEVICE_INLINE
    LocalBlockReader(const Word  *local_words,
                     unsigned int maxbits,
                     unsigned int local_block_idx)
    {
      size_t bit_offset = (size_t)local_block_idx * maxbits;
      m_words           = local_words + bit_offset / WSIZE;
      m_buffer          = m_words[0];
      m_current_bit     = (int)(bit_offset % WSIZE);
      m_buffer >>= m_current_bit;
    }

    COMPRESSION_DEVICE_INLINE
    unsigned int
    read_bit()
    {
      if (m_current_bit >= (int)WSIZE)
        {
          m_current_bit = 0;
          ++m_words;
          m_buffer = m_words[0];
        }
      unsigned int bit = m_buffer & 1;
      ++m_current_bit;
      m_buffer >>= 1;
      return bit;
    }

    COMPRESSION_DEVICE_INLINE
    uint64
    read_bits(unsigned int n_bits)
    {
      if (m_current_bit >= (int)WSIZE)
        {
          m_current_bit = 0;
          ++m_words;
          m_buffer = m_words[0];
        }
      int    rem_bits   = (int)WSIZE - m_current_bit;
      int    first_read = portable_min(rem_bits, (int)n_bits);
      Word   mask       = ((Word)1 << first_read) - 1;
      uint64 bits       = m_buffer & mask;
      m_buffer >>= first_read;
      m_current_bit += first_read;

      int next_read = 0;
      if ((int)n_bits > rem_bits)
        {
          ++m_words;
          m_buffer      = m_words[0];
          m_current_bit = 0;
          next_read     = (int)n_bits - first_read;
        }
      mask = ((Word)1 << next_read) - 1;
      bits += (m_buffer & mask) << first_read;
      m_buffer >>= next_read;
      m_current_bit += next_read;
      return bits;
    }
  };

  /* =========================================================================
     Device: InlineBlockWriter -- accumulate bits into a single uint64

     For bpt=1 specialized kernels: no Word array, no cross-word writes.
     The caller encodes one block, then stores the packed uint64 as
     uint32_t (8 bpv) or 3×uint16_t (12 bpv).
     =========================================================================
   */

  struct InlineBlockWriter
  {
    unsigned int m_current_bit;
    uint64       m_packed;

    COMPRESSION_DEVICE_INLINE
    InlineBlockWriter()
      : m_current_bit(0)
      , m_packed(0)
    {}

    COMPRESSION_DEVICE_INLINE
    uint64
    write_bits(uint64 bits, unsigned int n_bits)
    {
      if (n_bits == 0)
        return bits;
      uint64 b = n_bits < 64u ? (bits & (((uint64)1 << n_bits) - 1u)) : bits;
      m_packed |= (b << m_current_bit);
      m_current_bit += n_bits;
      return bits >> (uint64)n_bits;
    }

    COMPRESSION_DEVICE_INLINE
    unsigned int
    write_bit(unsigned int bit)
    {
      m_packed |= ((uint64)bit << m_current_bit);
      m_current_bit += 1;
      return bit;
    }
  };

  /* =========================================================================
     Device: InlineBlockReader -- read bits from a single uint64

     For bpt=1 specialized decompress kernels. No word-boundary crossing.
     =========================================================================
   */

  struct InlineBlockReader
  {
    uint64 m_buffer;

    COMPRESSION_DEVICE_INLINE
    InlineBlockReader(uint64 packed)
      : m_buffer(packed)
    {}

    COMPRESSION_DEVICE_INLINE
    unsigned int
    read_bit()
    {
      unsigned int bit = (unsigned int)(m_buffer & 1u);
      m_buffer >>= 1;
      return bit;
    }

    COMPRESSION_DEVICE_INLINE
    uint64
    read_bits(unsigned int n_bits)
    {
      uint64 mask = n_bits < 64u ? (((uint64)1 << n_bits) - 1u) : ~(uint64)0;
      uint64 bits = m_buffer & mask;
      m_buffer >>= n_bits;
      return bits;
    }
  };

  /* =========================================================================
     Device: BlockReader -- read bits from compressed stream (fixed-rate)

     Uses read-only load where available (CUDA __ldg).
     =========================================================================
   */

  struct BlockReader
  {
    int         m_current_bit;
    const Word *m_words;
    Word        m_buffer;

    COMPRESSION_DEVICE_INLINE
    BlockReader(const Word  *blocks,
                unsigned int maxbits,
                unsigned int block_idx)
    {
      size_t bit_offset = (size_t)block_idx * maxbits;
      size_t word_index = bit_offset / WSIZE;
      m_words           = blocks + word_index;
      m_buffer          = portable_ldg(m_words);
      m_current_bit     = (int)(bit_offset % WSIZE);
      m_buffer >>= m_current_bit;
    }

    COMPRESSION_DEVICE_INLINE
    unsigned int
    read_bit()
    {
      /* Advance-first: normalise if previous call left m_current_bit == WSIZE
       */
      if (m_current_bit >= (int)WSIZE)
        {
          m_current_bit = 0;
          ++m_words;
          m_buffer = portable_ldg(m_words);
        }
      unsigned int bit = m_buffer & 1;
      ++m_current_bit;
      m_buffer >>= 1;
      return bit;
    }

    COMPRESSION_DEVICE_INLINE
    uint64
    read_bits(unsigned int n_bits)
    {
      /* Advance-first: normalise if previous call left m_current_bit == WSIZE.
         This also guarantees rem_bits >= 1 so first_read <= 63 (no UB shift).
       */
      if (m_current_bit >= (int)WSIZE)
        {
          m_current_bit = 0;
          ++m_words;
          m_buffer = portable_ldg(m_words);
        }

      int    rem_bits   = (int)WSIZE - m_current_bit;
      int    first_read = portable_min(rem_bits, (int)n_bits);
      Word   mask       = ((Word)1 << first_read) - 1;
      uint64 bits       = m_buffer & mask;
      m_buffer >>= first_read;
      m_current_bit += first_read;

      int next_read = 0;
      /* Strict >: only advance to next word when bits actually spill over.
         Using >= would load word[num_words] (OOB) whenever n_bits == rem_bits.
       */
      if ((int)n_bits > rem_bits)
        {
          ++m_words;
          m_buffer      = portable_ldg(m_words);
          m_current_bit = 0;
          next_read     = (int)n_bits - first_read;
        }

      /* next_read <= n_bits - 1 <= 62, so shift is safe */
      mask = ((Word)1 << next_read) - 1;
      bits += (m_buffer & mask) << first_read;
      m_buffer >>= next_read;
      m_current_bit += next_read;
      return bits;
    }
  };

  /* =========================================================================
     Device: encode a 1D block of 4 values (templated on Writer)
     =========================================================================
   */

  template <typename Writer, typename Int, typename UInt>
  COMPRESSION_DEVICE_INLINE void
  encode_block_ints(Writer &writer, int maxbits, int maxprec, Int *iblock)
  {
    /* decorrelating transform */
    fwd_lift(iblock);

    /* reorder (identity for 1D) + signed -> unsigned negabinary */
    UInt ublock[4];
    for (int i = 0; i < 4; i++)
      ublock[i] = int2uint(iblock[i]);

    /* bit-plane encode */
    unsigned int intprec = (unsigned int)(CHAR_BIT * sizeof(UInt));
    unsigned int kmin =
      intprec > (unsigned int)maxprec ? intprec - (unsigned int)maxprec : 0u;
    unsigned int bits = (unsigned int)maxbits;

    for (unsigned int k = intprec, n = 0; bits && k-- > kmin;)
      {
        /* extract bit plane k (unrolled: eliminates loop-carried accumulation)
         */
        uint64 x = ((uint64)((ublock[0] >> k) & 1u)) |
                   (((uint64)((ublock[1] >> k) & 1u)) << 1) |
                   (((uint64)((ublock[2] >> k) & 1u)) << 2) |
                   (((uint64)((ublock[3] >> k) & 1u)) << 3);

        /* encode first n known bits */
        unsigned int m = portable_min(n, bits);
        bits -= m;
        x = writer.write_bits(x, m);

        /* run-length encode remainder */
        for (; n < 4 && bits && (bits--, writer.write_bit(!!x)); x >>= 1, n++)
          for (; n < 3 && bits && (bits--, !writer.write_bit(x & 1u));
               x >>= 1, n++)
            ;
      }
  }

  template <typename Writer, typename Scalar>
  COMPRESSION_DEVICE_INLINE void
  encode_block(Writer &writer, Scalar *fblock, unsigned int maxbits)
  {
    typedef typename traits<Scalar>::Int  Int;
    typedef typename traits<Scalar>::UInt UInt;

    int emax = max_exponent<Scalar>(fblock);
    int maxprec =
      calc_precision(emax, traits<Scalar>::PREC, traits<Scalar>::MINEXP);
    unsigned int e =
      maxprec ? (unsigned int)(emax + traits<Scalar>::EBIAS) : 0u;

    if (e)
      {
        const unsigned int ebits = (unsigned int)traits<Scalar>::EBITS + 1u;
        writer.write_bits(2u * e + 1u, ebits);

        Int iblock[4];
        fwd_cast<Scalar, Int>(iblock, fblock, emax);
        encode_block_ints<Writer, Int, UInt>(writer,
                                             (int)(maxbits - ebits),
                                             maxprec,
                                             iblock);
      }
    /* zero block: nothing written (local buffer was pre-zeroed) */
  }

  /* =========================================================================
     Device: decode a 1D block of 4 values
     =========================================================================
   */

  template <typename Reader, typename UInt>
  COMPRESSION_DEVICE_INLINE void
  decode_ints(Reader &reader, unsigned int maxbits, UInt *data)
  {
    const unsigned int intprec = (unsigned int)(CHAR_BIT * sizeof(UInt));
    unsigned int       bits    = maxbits;
    unsigned int       k, m, n;

    for (int i = 0; i < 4; i++)
      data[i] = 0;

    for (k = intprec, m = n = 0; bits && (m = 0, k-- > 0u);)
      {
        /* step 1: decode first n bits of bit plane k */
        m = portable_min(n, bits);
        bits -= m;
        uint64 x = reader.read_bits(m);

        /* step 2: unary run-length decode remainder */
        for (; bits && n < 4u; n++, m = n)
          {
            bits--;
            if (reader.read_bit())
              {
                for (; bits && n < 3u; n++)
                  {
                    bits--;
                    if (reader.read_bit())
                      break;
                  }
                x += (uint64)1 << n;
              }
            else
              {
                m = 4;
                break;
              }
          }

        /* step 3: deposit bit plane */
        for (unsigned int i = 0; i < 4; i++, x >>= 1)
          data[i] += (UInt)(x & 1u) << k;
      }
  }

  template <typename Reader, typename Scalar>
  COMPRESSION_DEVICE_INLINE void
  decode_block(Reader &reader, Scalar *fblock, unsigned int maxbits)
  {
    typedef typename traits<Scalar>::Int  Int;
    typedef typename traits<Scalar>::UInt UInt;

    unsigned int s_cont = reader.read_bit();
    if (!s_cont)
      {
        /* zero block */
        for (int i = 0; i < 4; i++)
          fblock[i] = (Scalar)0;
        return;
      }

    const unsigned int ebits = (unsigned int)traits<Scalar>::EBITS + 1u;
    int emax = (int)reader.read_bits(ebits - 1u) - traits<Scalar>::EBIAS;
    maxbits -= ebits;

    UInt ublock[4];
    decode_ints<Reader, UInt>(reader, maxbits, ublock);

    /* inverse reorder (identity for 1D) + unsigned -> signed */
    Int iblock[4];
    for (int i = 0; i < 4; i++)
      iblock[i] = uint2int(ublock[i]);

    inv_lift(iblock);

    Scalar inv_w = dequantize<Int, Scalar>(emax);
    for (int i = 0; i < 4; i++)
      fblock[i] = inv_w * (Scalar)iblock[i];
  }

  /* =========================================================================
     GPU kernels: ZFP super-block (fallback for bpv other than 8 and 12)

     Each thread processes 'bpt' blocks whose combined output fills exactly
     'wpt' complete 64-bit Words. Bits are accumulated in a thread-local
     buffer (LocalBlockWriter with |=), then flushed to global memory with
     direct stores. No atomics, no memset.

     For 8 bpv and 12 bpv, specialized bpt=1 kernels (compress_zfp_32_kernel,
     compress_zfp_48_kernel, etc.) are used instead — see below.

     Parameters computed on host:
       g   = gcd(maxbits, 64)
       bpt = 64 / g          (blocks per thread)
       wpt = maxbits / g     (words per thread)
     Guarantee: bpt * maxbits = wpt * 64 (word-aligned)

     Example:
       16 bpv: maxbits=64,  bpt=1,  wpt=1  (1 block  → 1 word)
     =========================================================================
   */

  static constexpr int COMPRESSION_BLOCK_SIZE   = 256;
  static constexpr int MAX_WORDS_PER_SUPERBLOCK = 16;

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)

  template <typename Scalar>
  COMPRESSION_GLOBAL void
  compress_kernel(const Scalar *COMPRESSION_RESTRICT data,
                  Word *COMPRESSION_RESTRICT         stream,
                  unsigned int                       maxbits,
                  unsigned int                       dim,
                  unsigned int                       tot_blocks,
                  unsigned int                       bpt,
                  unsigned int                       wpt,
                  unsigned int                       num_words)
  {
    const unsigned int super_idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* thread-local output buffer, zero-initialized */
    Word local_words[MAX_WORDS_PER_SUPERBLOCK];
    for (unsigned int w = 0; w < wpt; w++)
      local_words[w] = 0;

    /* process bpt blocks sequentially into local buffer */
    const unsigned int block_base = super_idx * bpt;
    for (unsigned int b = 0; b < bpt; b++)
      {
        unsigned int block_idx = block_base + b;
        if (block_idx >= tot_blocks)
          break;

        unsigned int block_start = block_idx * 4u;
        Scalar       fblock[4];

        if (block_start + 4u <= dim)
          {
            fblock[0] = data[block_start];
            fblock[1] = data[block_start + 1];
            fblock[2] = data[block_start + 2];
            fblock[3] = data[block_start + 3];
          }
        else
          {
            unsigned int nx = dim - block_start;
            for (unsigned int i = 0; i < nx; i++)
              fblock[i] = data[block_start + i];
            pad_block(fblock, nx);
          }

        LocalBlockWriter writer(local_words, maxbits, b);
        encode_block(writer, fblock, maxbits);
      }

    /* flush local buffer to global stream (direct store, no atomics) */
    size_t global_word_base = (size_t)super_idx * wpt;
    for (unsigned int w = 0; w < wpt; w++)
      {
        size_t gw = global_word_base + w;
        if (gw < num_words)
          stream[gw] = local_words[w];
      }
  }

  /* Super-block decompress: each thread loads wpt Words into a local buffer,
     then decodes bpt blocks using LocalBlockReader (L1-cached reads).
     This mirrors the compress super-block kernel for bandwidth symmetry. */
  template <typename Scalar>
  COMPRESSION_GLOBAL void
  decompress_kernel_sb(const Word *COMPRESSION_RESTRICT stream,
                       Scalar *COMPRESSION_RESTRICT     data,
                       unsigned int                     maxbits,
                       unsigned int                     dim,
                       unsigned int                     tot_blocks,
                       unsigned int                     bpt,
                       unsigned int                     wpt,
                       unsigned int                     num_words)
  {
    const unsigned int super_idx  = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int block_base = super_idx * bpt;
    if (block_base >= tot_blocks)
      return;

    /* Load wpt words from global stream into thread-local buffer */
    Word   local_words[MAX_WORDS_PER_SUPERBLOCK];
    size_t global_word_base = (size_t)super_idx * wpt;
    for (unsigned int w = 0; w < wpt; w++)
      {
        size_t gw      = global_word_base + w;
        local_words[w] = (gw < num_words) ? portable_ldg(stream + gw) : (Word)0;
      }

    /* Decode bpt blocks from the local buffer */
    for (unsigned int b = 0; b < bpt; b++)
      {
        unsigned int block_idx = block_base + b;
        if (block_idx >= tot_blocks)
          break;

        LocalBlockReader reader(local_words, maxbits, b);
        Scalar           fblock[4];
        decode_block<LocalBlockReader, Scalar>(reader, fblock, maxbits);

        const unsigned int block_start = block_idx * 4u;
        if (block_start + 4u <= dim)
          {
            data[block_start]     = fblock[0];
            data[block_start + 1] = fblock[1];
            data[block_start + 2] = fblock[2];
            data[block_start + 3] = fblock[3];
          }
        else
          {
            unsigned int nx = dim - block_start;
            for (unsigned int i = 0; i < nx; i++)
              data[block_start + i] = fblock[i];
          }
      }
  }

  /* -----------------------------------------------------------------------
     Fused gather+compress kernel (CUDA/HIP)

     Reads scattered data via indirection and compresses directly to the
     output stream. Eliminates the intermediate full-precision send buffer.

     Assumes gatherBlockSize is a multiple of 4 (true when blockSize =
     numWaveFunctions, which is always a multiple of 4). This means each
     ZFP block of 4 values falls entirely within one index entry, allowing
     a single index lookup per block and contiguous 4-element reads.
     ----------------------------------------------------------------------- */
  template <typename Scalar, typename IndexType>
  COMPRESSION_GLOBAL void
  compress_gather_kernel(const Scalar     *COMPRESSION_RESTRICT dataArray,
                         const IndexType  *COMPRESSION_RESTRICT indices,
                         unsigned int                           gatherBlockSize,
                         Word *COMPRESSION_RESTRICT               stream,
                         unsigned int                             maxbits,
                         unsigned int                             tot_blocks,
                         unsigned int                             bpt,
                         unsigned int                             wpt,
                         unsigned int                             num_words)
  {
    const unsigned int super_idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* blocks_per_entry = gatherBlockSize / 4 (exact since gatherBlockSize
       is a multiple of 4). This lets us convert ZFP block_idx directly to
       (index entry, offset within entry) with one division. */
    const unsigned int blocks_per_entry = gatherBlockSize >> 2;

    Word local_words[MAX_WORDS_PER_SUPERBLOCK];
    for (unsigned int w = 0; w < wpt; w++)
      local_words[w] = 0;

    const unsigned int block_base = super_idx * bpt;
    for (unsigned int b = 0; b < bpt; b++)
      {
        unsigned int block_idx = block_base + b;
        if (block_idx >= tot_blocks)
          break;

        /* Single index lookup: which index entry does this ZFP block belong
           to, and what is the offset (in units of 4 values) within that
           entry? */
        unsigned int gatherIdx = block_idx / blocks_per_entry;
        unsigned int localBlock = block_idx - gatherIdx * blocks_per_entry;
        unsigned int intraIdx = localBlock * 4u;

        /* Base address in dataArray for this ZFP block — contiguous read.
           blockSize is always a multiple of 4, so every block is full. */
        size_t base =
          (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

        Scalar fblock[4];
        fblock[0] = dataArray[base];
        fblock[1] = dataArray[base + 1];
        fblock[2] = dataArray[base + 2];
        fblock[3] = dataArray[base + 3];

        LocalBlockWriter writer(local_words, maxbits, b);
        encode_block(writer, fblock, maxbits);
      }

    size_t global_word_base = (size_t)super_idx * wpt;
    for (unsigned int w = 0; w < wpt; w++)
      {
        size_t gw = global_word_base + w;
        if (gw < num_words)
          stream[gw] = local_words[w];
      }
  }

  /* -----------------------------------------------------------------------
     Fused decompress+scatter_add kernel (CUDA/HIP)

     Decompresses and atomicAdds directly to scattered positions in dataArray.
     Eliminates the intermediate full-precision recv buffer.

     Assumes gatherBlockSize is a multiple of 4, so every ZFP block is full
     (no partial blocks).
     ----------------------------------------------------------------------- */
  template <typename Scalar, typename IndexType>
  COMPRESSION_GLOBAL void
  decompress_scatter_add_kernel(
    const Word      *COMPRESSION_RESTRICT stream,
    const IndexType *COMPRESSION_RESTRICT indices,
    unsigned int                          gatherBlockSize,
    Scalar *COMPRESSION_RESTRICT             dataArray,
    unsigned int                             maxbits,
    unsigned int                             tot_blocks,
    unsigned int                             bpt,
    unsigned int                             wpt,
    unsigned int                             num_words)
  {
    const unsigned int super_idx  = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int block_base = super_idx * bpt;
    if (block_base >= tot_blocks)
      return;

    const unsigned int blocks_per_entry = gatherBlockSize >> 2;

    Word   local_words[MAX_WORDS_PER_SUPERBLOCK];
    size_t global_word_base = (size_t)super_idx * wpt;
    for (unsigned int w = 0; w < wpt; w++)
      {
        size_t gw      = global_word_base + w;
        local_words[w] = (gw < num_words) ? portable_ldg(stream + gw) : (Word)0;
      }

    for (unsigned int b = 0; b < bpt; b++)
      {
        unsigned int block_idx = block_base + b;
        if (block_idx >= tot_blocks)
          break;

        LocalBlockReader reader(local_words, maxbits, b);
        Scalar           fblock[4];
        decode_block<LocalBlockReader, Scalar>(reader, fblock, maxbits);

        unsigned int gatherIdx  = block_idx / blocks_per_entry;
        unsigned int localBlock = block_idx - gatherIdx * blocks_per_entry;
        unsigned int intraIdx   = localBlock * 4u;
        size_t base =
          (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

        portable_atomicAdd(&dataArray[base],     fblock[0]);
        portable_atomicAdd(&dataArray[base + 1], fblock[1]);
        portable_atomicAdd(&dataArray[base + 2], fblock[2]);
        portable_atomicAdd(&dataArray[base + 3], fblock[3]);
      }
  }

#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)

  template <typename Scalar>
  void
  compress_kernel(sycl::nd_item<1> item,
                  const Scalar    *data,
                  Word            *stream,
                  unsigned int     maxbits,
                  unsigned int     dim,
                  unsigned int     tot_blocks,
                  unsigned int     bpt,
                  unsigned int     wpt,
                  unsigned int     num_words)
  {
    const unsigned int super_idx = item.get_global_id(0);

    Word local_words[MAX_WORDS_PER_SUPERBLOCK];
    for (unsigned int w = 0; w < wpt; w++)
      local_words[w] = 0;

    const unsigned int block_base = super_idx * bpt;
    for (unsigned int b = 0; b < bpt; b++)
      {
        unsigned int block_idx = block_base + b;
        if (block_idx >= tot_blocks)
          break;

        unsigned int block_start = block_idx * 4u;
        Scalar       fblock[4];

        if (block_start + 4u <= dim)
          {
            fblock[0] = data[block_start];
            fblock[1] = data[block_start + 1];
            fblock[2] = data[block_start + 2];
            fblock[3] = data[block_start + 3];
          }
        else
          {
            unsigned int nx = dim - block_start;
            for (unsigned int i = 0; i < nx; i++)
              fblock[i] = data[block_start + i];
            pad_block(fblock, nx);
          }

        LocalBlockWriter writer(local_words, maxbits, b);
        encode_block(writer, fblock, maxbits);
      }

    size_t global_word_base = (size_t)super_idx * wpt;
    for (unsigned int w = 0; w < wpt; w++)
      {
        size_t gw = global_word_base + w;
        if (gw < num_words)
          stream[gw] = local_words[w];
      }
  }

  template <typename Scalar>
  void
  decompress_kernel(sycl::nd_item<1> item,
                    const Word      *stream,
                    Scalar          *data,
                    unsigned int     maxbits,
                    unsigned int     dim,
                    unsigned int     tot_blocks)
  {
    const unsigned int block_idx = item.get_global_id(0);
    if (block_idx >= tot_blocks)
      return;

    BlockReader reader(stream, maxbits, block_idx);
    Scalar      fblock[4];
    decode_block<BlockReader, Scalar>(reader, fblock, maxbits);

    const unsigned int block_start = block_idx * 4u;
    if (block_start + 4u <= dim)
      {
        data[block_start]     = fblock[0];
        data[block_start + 1] = fblock[1];
        data[block_start + 2] = fblock[2];
        data[block_start + 3] = fblock[3];
      }
    else
      {
        unsigned int nx = dim - block_start;
        for (unsigned int i = 0; i < nx; i++)
          data[block_start + i] = fblock[i];
      }
  }

  template <typename Scalar>
  void
  decompress_kernel_sb(sycl::nd_item<1> item,
                       const Word      *stream,
                       Scalar          *data,
                       unsigned int     maxbits,
                       unsigned int     dim,
                       unsigned int     tot_blocks,
                       unsigned int     bpt,
                       unsigned int     wpt,
                       unsigned int     num_words)
  {
    const unsigned int super_idx  = item.get_global_id(0);
    const unsigned int block_base = super_idx * bpt;
    if (block_base >= tot_blocks)
      return;

    Word   local_words[MAX_WORDS_PER_SUPERBLOCK];
    size_t global_word_base = (size_t)super_idx * wpt;
    for (unsigned int w = 0; w < wpt; w++)
      {
        size_t gw      = global_word_base + w;
        local_words[w] = (gw < num_words) ? stream[gw] : (Word)0;
      }

    for (unsigned int b = 0; b < bpt; b++)
      {
        unsigned int block_idx = block_base + b;
        if (block_idx >= tot_blocks)
          break;

        LocalBlockReader reader(local_words, maxbits, b);
        Scalar           fblock[4];
        decode_block<LocalBlockReader, Scalar>(reader, fblock, maxbits);

        const unsigned int block_start = block_idx * 4u;
        if (block_start + 4u <= dim)
          {
            data[block_start]     = fblock[0];
            data[block_start + 1] = fblock[1];
            data[block_start + 2] = fblock[2];
            data[block_start + 3] = fblock[3];
          }
        else
          {
            unsigned int nx = dim - block_start;
            for (unsigned int i = 0; i < nx; i++)
              data[block_start + i] = fblock[i];
          }
      }
  }

  /* -----------------------------------------------------------------------
     Fused gather+compress kernel (SYCL)
     ----------------------------------------------------------------------- */
  template <typename Scalar, typename IndexType>
  void
  compress_gather_kernel(sycl::nd_item<1>   item,
                         const Scalar      *dataArray,
                         const IndexType   *indices,
                         unsigned int        gatherBlockSize,
                         Word               *stream,
                         unsigned int        maxbits,
                         unsigned int        tot_blocks,
                         unsigned int        bpt,
                         unsigned int        wpt,
                         unsigned int        num_words)
  {
    const unsigned int super_idx = item.get_global_id(0);

    /* blocks_per_entry = gatherBlockSize / 4 (exact since gatherBlockSize
       is a multiple of 4). One division per ZFP block instead of per element. */
    const unsigned int blocks_per_entry = gatherBlockSize >> 2;

    Word local_words[MAX_WORDS_PER_SUPERBLOCK];
    for (unsigned int w = 0; w < wpt; w++)
      local_words[w] = 0;

    const unsigned int block_base = super_idx * bpt;
    for (unsigned int b = 0; b < bpt; b++)
      {
        unsigned int block_idx = block_base + b;
        if (block_idx >= tot_blocks)
          break;

        /* Single index lookup per ZFP block — contiguous read.
           blockSize is always a multiple of 4, so every block is full. */
        unsigned int gatherIdx  = block_idx / blocks_per_entry;
        unsigned int localBlock = block_idx - gatherIdx * blocks_per_entry;
        unsigned int intraIdx   = localBlock * 4u;
        size_t base =
          (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

        Scalar fblock[4];
        fblock[0] = dataArray[base];
        fblock[1] = dataArray[base + 1];
        fblock[2] = dataArray[base + 2];
        fblock[3] = dataArray[base + 3];

        LocalBlockWriter writer(local_words, maxbits, b);
        encode_block(writer, fblock, maxbits);
      }

    size_t global_word_base = (size_t)super_idx * wpt;
    for (unsigned int w = 0; w < wpt; w++)
      {
        size_t gw = global_word_base + w;
        if (gw < num_words)
          stream[gw] = local_words[w];
      }
  }

  /* -----------------------------------------------------------------------
     Fused decompress+scatter_add kernel (SYCL)

     Assumes gatherBlockSize is a multiple of 4, so every ZFP block is full.
     ----------------------------------------------------------------------- */
  template <typename Scalar, typename IndexType>
  void
  decompress_scatter_add_kernel(sycl::nd_item<1>   item,
                                const Word        *stream,
                                const IndexType   *indices,
                                unsigned int        gatherBlockSize,
                                Scalar             *dataArray,
                                unsigned int        maxbits,
                                unsigned int        tot_blocks,
                                unsigned int        bpt,
                                unsigned int        wpt,
                                unsigned int        num_words)
  {
    const unsigned int super_idx  = item.get_global_id(0);
    const unsigned int block_base = super_idx * bpt;
    if (block_base >= tot_blocks)
      return;

    const unsigned int blocks_per_entry = gatherBlockSize >> 2;

    Word   local_words[MAX_WORDS_PER_SUPERBLOCK];
    size_t global_word_base = (size_t)super_idx * wpt;
    for (unsigned int w = 0; w < wpt; w++)
      {
        size_t gw      = global_word_base + w;
        local_words[w] = (gw < num_words) ? stream[gw] : (Word)0;
      }

    for (unsigned int b = 0; b < bpt; b++)
      {
        unsigned int block_idx = block_base + b;
        if (block_idx >= tot_blocks)
          break;

        LocalBlockReader reader(local_words, maxbits, b);
        Scalar           fblock[4];
        decode_block<LocalBlockReader, Scalar>(reader, fblock, maxbits);

        unsigned int gatherIdx  = block_idx / blocks_per_entry;
        unsigned int localBlock = block_idx - gatherIdx * blocks_per_entry;
        unsigned int intraIdx   = localBlock * 4u;
        size_t base =
          (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

        portable_atomicAdd(&dataArray[base],     fblock[0]);
        portable_atomicAdd(&dataArray[base + 1], fblock[1]);
        portable_atomicAdd(&dataArray[base + 2], fblock[2]);
        portable_atomicAdd(&dataArray[base + 3], fblock[3]);
      }
  }

#endif /* DFTFE_WITH_DEVICE_LANG_SYCL */

  /* =========================================================================
     Specialized ZFP 8-bpv (32-bit) and 12-bpv (48-bit) kernels

     1 thread per block, eliminates the super-block mechanism for 8 and 12 bpv.
     Uses InlineBlockWriter/Reader instead of LocalBlockWriter/Reader + Word
     arrays. Maximizes GPU thread count for small problem sizes.

       8 bpv (maxbits=32): each block → 1 × uint32_t   (bpt=1)
      12 bpv (maxbits=48): each block → 3 × uint16_t   (bpt=1)

     Memory layout is compatible with compressed_size() allocation.
     On little-endian GPUs, the 8bpv uint32_t layout is byte-identical to
     the super-block Word layout. The 12bpv uint16_t layout differs from
     the super-block Word layout but compress/decompress are always paired.
     =========================================================================
   */

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)

  /* --- ZFP 8bpv: 1 thread/block, 1 × uint32_t --- */

  template <typename Scalar>
  COMPRESSION_GLOBAL void
  compress_zfp_32_kernel(const Scalar *COMPRESSION_RESTRICT data,
                         unsigned int *COMPRESSION_RESTRICT stream,
                         unsigned int                       dim,
                         unsigned int                       tot_blocks)
  {
    const unsigned int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= tot_blocks)
      return;

    unsigned int block_start = block_idx * 4u;
    Scalar       fblock[4];

    if (block_start + 4u <= dim)
      {
        fblock[0] = data[block_start];
        fblock[1] = data[block_start + 1];
        fblock[2] = data[block_start + 2];
        fblock[3] = data[block_start + 3];
      }
    else
      {
        unsigned int nx = dim - block_start;
        for (unsigned int j = 0; j < nx; j++)
          fblock[j] = data[block_start + j];
        pad_block(fblock, nx);
      }

    InlineBlockWriter writer;
    encode_block(writer, fblock, 32u);
    stream[block_idx] = (unsigned int)writer.m_packed;
  }

  template <typename Scalar>
  COMPRESSION_GLOBAL void
  decompress_zfp_32_kernel(const unsigned int *COMPRESSION_RESTRICT stream,
                           Scalar *COMPRESSION_RESTRICT             data,
                           unsigned int                             dim,
                           unsigned int                             tot_blocks)
  {
    const unsigned int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= tot_blocks)
      return;

    InlineBlockReader reader((uint64)stream[block_idx]);
    Scalar            fblock[4];
    decode_block<InlineBlockReader, Scalar>(reader, fblock, 32u);

    unsigned int block_start = block_idx * 4u;
    if (block_start + 4u <= dim)
      {
        data[block_start]     = fblock[0];
        data[block_start + 1] = fblock[1];
        data[block_start + 2] = fblock[2];
        data[block_start + 3] = fblock[3];
      }
    else
      {
        unsigned int nx = dim - block_start;
        for (unsigned int j = 0; j < nx; j++)
          data[block_start + j] = fblock[j];
      }
  }

  template <typename Scalar, typename IndexType>
  COMPRESSION_GLOBAL void
  compress_gather_zfp_32_kernel(
    const Scalar    *COMPRESSION_RESTRICT dataArray,
    const IndexType *COMPRESSION_RESTRICT indices,
    unsigned int                          gatherBlockSize,
    unsigned int *COMPRESSION_RESTRICT    stream,
    unsigned int                          tot_blocks)
  {
    const unsigned int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= tot_blocks)
      return;

    const unsigned int blocks_per_entry = gatherBlockSize >> 2;
    unsigned int       gatherIdx  = block_idx / blocks_per_entry;
    unsigned int       localBlock = block_idx - gatherIdx * blocks_per_entry;
    unsigned int       intraIdx   = localBlock * 4u;
    size_t base =
      (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

    Scalar fblock[4];
    fblock[0] = dataArray[base];
    fblock[1] = dataArray[base + 1];
    fblock[2] = dataArray[base + 2];
    fblock[3] = dataArray[base + 3];

    InlineBlockWriter writer;
    encode_block(writer, fblock, 32u);
    stream[block_idx] = (unsigned int)writer.m_packed;
  }

  template <typename Scalar, typename IndexType>
  COMPRESSION_GLOBAL void
  decompress_scatter_add_zfp_32_kernel(
    const unsigned int *COMPRESSION_RESTRICT stream,
    const IndexType    *COMPRESSION_RESTRICT indices,
    unsigned int                             gatherBlockSize,
    Scalar *COMPRESSION_RESTRICT             dataArray,
    unsigned int                             tot_blocks)
  {
    const unsigned int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= tot_blocks)
      return;

    InlineBlockReader reader((uint64)stream[block_idx]);
    Scalar            fblock[4];
    decode_block<InlineBlockReader, Scalar>(reader, fblock, 32u);

    const unsigned int blocks_per_entry = gatherBlockSize >> 2;
    unsigned int       gatherIdx  = block_idx / blocks_per_entry;
    unsigned int       localBlock = block_idx - gatherIdx * blocks_per_entry;
    unsigned int       intraIdx   = localBlock * 4u;
    size_t base =
      (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

    portable_atomicAdd(&dataArray[base],     fblock[0]);
    portable_atomicAdd(&dataArray[base + 1], fblock[1]);
    portable_atomicAdd(&dataArray[base + 2], fblock[2]);
    portable_atomicAdd(&dataArray[base + 3], fblock[3]);
  }

  /* --- ZFP 12bpv: 1 thread/block, 3 × uint16_t --- */

  template <typename Scalar>
  COMPRESSION_GLOBAL void
  compress_zfp_48_kernel(const Scalar *COMPRESSION_RESTRICT data,
                         uint16_t *COMPRESSION_RESTRICT     stream,
                         unsigned int                       dim,
                         unsigned int                       tot_blocks)
  {
    const unsigned int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= tot_blocks)
      return;

    unsigned int block_start = block_idx * 4u;
    Scalar       fblock[4];

    if (block_start + 4u <= dim)
      {
        fblock[0] = data[block_start];
        fblock[1] = data[block_start + 1];
        fblock[2] = data[block_start + 2];
        fblock[3] = data[block_start + 3];
      }
    else
      {
        unsigned int nx = dim - block_start;
        for (unsigned int j = 0; j < nx; j++)
          fblock[j] = data[block_start + j];
        pad_block(fblock, nx);
      }

    InlineBlockWriter writer;
    encode_block(writer, fblock, 48u);
    size_t out    = (size_t)block_idx * 3u;
    stream[out]     = (uint16_t)(writer.m_packed);
    stream[out + 1] = (uint16_t)(writer.m_packed >> 16);
    stream[out + 2] = (uint16_t)(writer.m_packed >> 32);
  }

  template <typename Scalar>
  COMPRESSION_GLOBAL void
  decompress_zfp_48_kernel(const uint16_t *COMPRESSION_RESTRICT stream,
                           Scalar *COMPRESSION_RESTRICT         data,
                           unsigned int                         dim,
                           unsigned int                         tot_blocks)
  {
    const unsigned int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= tot_blocks)
      return;

    size_t out    = (size_t)block_idx * 3u;
    uint64 packed = (uint64)stream[out] |
                    ((uint64)stream[out + 1] << 16) |
                    ((uint64)stream[out + 2] << 32);

    InlineBlockReader reader(packed);
    Scalar            fblock[4];
    decode_block<InlineBlockReader, Scalar>(reader, fblock, 48u);

    unsigned int block_start = block_idx * 4u;
    if (block_start + 4u <= dim)
      {
        data[block_start]     = fblock[0];
        data[block_start + 1] = fblock[1];
        data[block_start + 2] = fblock[2];
        data[block_start + 3] = fblock[3];
      }
    else
      {
        unsigned int nx = dim - block_start;
        for (unsigned int j = 0; j < nx; j++)
          data[block_start + j] = fblock[j];
      }
  }

  template <typename Scalar, typename IndexType>
  COMPRESSION_GLOBAL void
  compress_gather_zfp_48_kernel(
    const Scalar    *COMPRESSION_RESTRICT dataArray,
    const IndexType *COMPRESSION_RESTRICT indices,
    unsigned int                          gatherBlockSize,
    uint16_t *COMPRESSION_RESTRICT        stream,
    unsigned int                          tot_blocks)
  {
    const unsigned int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= tot_blocks)
      return;

    const unsigned int blocks_per_entry = gatherBlockSize >> 2;
    unsigned int       gatherIdx  = block_idx / blocks_per_entry;
    unsigned int       localBlock = block_idx - gatherIdx * blocks_per_entry;
    unsigned int       intraIdx   = localBlock * 4u;
    size_t base =
      (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

    Scalar fblock[4];
    fblock[0] = dataArray[base];
    fblock[1] = dataArray[base + 1];
    fblock[2] = dataArray[base + 2];
    fblock[3] = dataArray[base + 3];

    InlineBlockWriter writer;
    encode_block(writer, fblock, 48u);
    size_t out    = (size_t)block_idx * 3u;
    stream[out]     = (uint16_t)(writer.m_packed);
    stream[out + 1] = (uint16_t)(writer.m_packed >> 16);
    stream[out + 2] = (uint16_t)(writer.m_packed >> 32);
  }

  template <typename Scalar, typename IndexType>
  COMPRESSION_GLOBAL void
  decompress_scatter_add_zfp_48_kernel(
    const uint16_t  *COMPRESSION_RESTRICT stream,
    const IndexType *COMPRESSION_RESTRICT indices,
    unsigned int                          gatherBlockSize,
    Scalar *COMPRESSION_RESTRICT          dataArray,
    unsigned int                          tot_blocks)
  {
    const unsigned int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= tot_blocks)
      return;

    size_t out    = (size_t)block_idx * 3u;
    uint64 packed = (uint64)stream[out] |
                    ((uint64)stream[out + 1] << 16) |
                    ((uint64)stream[out + 2] << 32);

    InlineBlockReader reader(packed);
    Scalar            fblock[4];
    decode_block<InlineBlockReader, Scalar>(reader, fblock, 48u);

    const unsigned int blocks_per_entry = gatherBlockSize >> 2;
    unsigned int       gatherIdx  = block_idx / blocks_per_entry;
    unsigned int       localBlock = block_idx - gatherIdx * blocks_per_entry;
    unsigned int       intraIdx   = localBlock * 4u;
    size_t base =
      (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

    portable_atomicAdd(&dataArray[base],     fblock[0]);
    portable_atomicAdd(&dataArray[base + 1], fblock[1]);
    portable_atomicAdd(&dataArray[base + 2], fblock[2]);
    portable_atomicAdd(&dataArray[base + 3], fblock[3]);
  }

#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)

  /* --- ZFP 8bpv SYCL: 1 thread/block, 1 × uint32_t --- */

  template <typename Scalar>
  void
  compress_zfp_32_kernel(sycl::nd_item<1> item,
                         const Scalar    *data,
                         unsigned int    *stream,
                         unsigned int     dim,
                         unsigned int     tot_blocks)
  {
    const unsigned int block_idx = item.get_global_id(0);
    if (block_idx >= tot_blocks)
      return;

    unsigned int block_start = block_idx * 4u;
    Scalar       fblock[4];

    if (block_start + 4u <= dim)
      {
        fblock[0] = data[block_start];
        fblock[1] = data[block_start + 1];
        fblock[2] = data[block_start + 2];
        fblock[3] = data[block_start + 3];
      }
    else
      {
        unsigned int nx = dim - block_start;
        for (unsigned int j = 0; j < nx; j++)
          fblock[j] = data[block_start + j];
        pad_block(fblock, nx);
      }

    InlineBlockWriter writer;
    encode_block(writer, fblock, 32u);
    stream[block_idx] = (unsigned int)writer.m_packed;
  }

  template <typename Scalar>
  void
  decompress_zfp_32_kernel(sycl::nd_item<1>    item,
                           const unsigned int *stream,
                           Scalar             *data,
                           unsigned int        dim,
                           unsigned int        tot_blocks)
  {
    const unsigned int block_idx = item.get_global_id(0);
    if (block_idx >= tot_blocks)
      return;

    InlineBlockReader reader((uint64)stream[block_idx]);
    Scalar            fblock[4];
    decode_block<InlineBlockReader, Scalar>(reader, fblock, 32u);

    unsigned int block_start = block_idx * 4u;
    if (block_start + 4u <= dim)
      {
        data[block_start]     = fblock[0];
        data[block_start + 1] = fblock[1];
        data[block_start + 2] = fblock[2];
        data[block_start + 3] = fblock[3];
      }
    else
      {
        unsigned int nx = dim - block_start;
        for (unsigned int j = 0; j < nx; j++)
          data[block_start + j] = fblock[j];
      }
  }

  template <typename Scalar, typename IndexType>
  void
  compress_gather_zfp_32_kernel(sycl::nd_item<1>   item,
                                const Scalar      *dataArray,
                                const IndexType   *indices,
                                unsigned int        gatherBlockSize,
                                unsigned int       *stream,
                                unsigned int        tot_blocks)
  {
    const unsigned int block_idx = item.get_global_id(0);
    if (block_idx >= tot_blocks)
      return;

    const unsigned int blocks_per_entry = gatherBlockSize >> 2;
    unsigned int       gatherIdx  = block_idx / blocks_per_entry;
    unsigned int       localBlock = block_idx - gatherIdx * blocks_per_entry;
    unsigned int       intraIdx   = localBlock * 4u;
    size_t base =
      (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

    Scalar fblock[4];
    fblock[0] = dataArray[base];
    fblock[1] = dataArray[base + 1];
    fblock[2] = dataArray[base + 2];
    fblock[3] = dataArray[base + 3];

    InlineBlockWriter writer;
    encode_block(writer, fblock, 32u);
    stream[block_idx] = (unsigned int)writer.m_packed;
  }

  template <typename Scalar, typename IndexType>
  void
  decompress_scatter_add_zfp_32_kernel(
    sycl::nd_item<1>    item,
    const unsigned int *stream,
    const IndexType    *indices,
    unsigned int        gatherBlockSize,
    Scalar             *dataArray,
    unsigned int        tot_blocks)
  {
    const unsigned int block_idx = item.get_global_id(0);
    if (block_idx >= tot_blocks)
      return;

    InlineBlockReader reader((uint64)stream[block_idx]);
    Scalar            fblock[4];
    decode_block<InlineBlockReader, Scalar>(reader, fblock, 32u);

    const unsigned int blocks_per_entry = gatherBlockSize >> 2;
    unsigned int       gatherIdx  = block_idx / blocks_per_entry;
    unsigned int       localBlock = block_idx - gatherIdx * blocks_per_entry;
    unsigned int       intraIdx   = localBlock * 4u;
    size_t base =
      (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

    portable_atomicAdd(&dataArray[base],     fblock[0]);
    portable_atomicAdd(&dataArray[base + 1], fblock[1]);
    portable_atomicAdd(&dataArray[base + 2], fblock[2]);
    portable_atomicAdd(&dataArray[base + 3], fblock[3]);
  }

  /* --- ZFP 12bpv SYCL: 1 thread/block, 3 × uint16_t --- */

  template <typename Scalar>
  void
  compress_zfp_48_kernel(sycl::nd_item<1> item,
                         const Scalar    *data,
                         uint16_t        *stream,
                         unsigned int     dim,
                         unsigned int     tot_blocks)
  {
    const unsigned int block_idx = item.get_global_id(0);
    if (block_idx >= tot_blocks)
      return;

    unsigned int block_start = block_idx * 4u;
    Scalar       fblock[4];

    if (block_start + 4u <= dim)
      {
        fblock[0] = data[block_start];
        fblock[1] = data[block_start + 1];
        fblock[2] = data[block_start + 2];
        fblock[3] = data[block_start + 3];
      }
    else
      {
        unsigned int nx = dim - block_start;
        for (unsigned int j = 0; j < nx; j++)
          fblock[j] = data[block_start + j];
        pad_block(fblock, nx);
      }

    InlineBlockWriter writer;
    encode_block(writer, fblock, 48u);
    size_t out    = (size_t)block_idx * 3u;
    stream[out]     = (uint16_t)(writer.m_packed);
    stream[out + 1] = (uint16_t)(writer.m_packed >> 16);
    stream[out + 2] = (uint16_t)(writer.m_packed >> 32);
  }

  template <typename Scalar>
  void
  decompress_zfp_48_kernel(sycl::nd_item<1>  item,
                           const uint16_t   *stream,
                           Scalar           *data,
                           unsigned int      dim,
                           unsigned int      tot_blocks)
  {
    const unsigned int block_idx = item.get_global_id(0);
    if (block_idx >= tot_blocks)
      return;

    size_t out    = (size_t)block_idx * 3u;
    uint64 packed = (uint64)stream[out] |
                    ((uint64)stream[out + 1] << 16) |
                    ((uint64)stream[out + 2] << 32);

    InlineBlockReader reader(packed);
    Scalar            fblock[4];
    decode_block<InlineBlockReader, Scalar>(reader, fblock, 48u);

    unsigned int block_start = block_idx * 4u;
    if (block_start + 4u <= dim)
      {
        data[block_start]     = fblock[0];
        data[block_start + 1] = fblock[1];
        data[block_start + 2] = fblock[2];
        data[block_start + 3] = fblock[3];
      }
    else
      {
        unsigned int nx = dim - block_start;
        for (unsigned int j = 0; j < nx; j++)
          data[block_start + j] = fblock[j];
      }
  }

  template <typename Scalar, typename IndexType>
  void
  compress_gather_zfp_48_kernel(sycl::nd_item<1>  item,
                                const Scalar     *dataArray,
                                const IndexType  *indices,
                                unsigned int      gatherBlockSize,
                                uint16_t         *stream,
                                unsigned int      tot_blocks)
  {
    const unsigned int block_idx = item.get_global_id(0);
    if (block_idx >= tot_blocks)
      return;

    const unsigned int blocks_per_entry = gatherBlockSize >> 2;
    unsigned int       gatherIdx  = block_idx / blocks_per_entry;
    unsigned int       localBlock = block_idx - gatherIdx * blocks_per_entry;
    unsigned int       intraIdx   = localBlock * 4u;
    size_t base =
      (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

    Scalar fblock[4];
    fblock[0] = dataArray[base];
    fblock[1] = dataArray[base + 1];
    fblock[2] = dataArray[base + 2];
    fblock[3] = dataArray[base + 3];

    InlineBlockWriter writer;
    encode_block(writer, fblock, 48u);
    size_t out    = (size_t)block_idx * 3u;
    stream[out]     = (uint16_t)(writer.m_packed);
    stream[out + 1] = (uint16_t)(writer.m_packed >> 16);
    stream[out + 2] = (uint16_t)(writer.m_packed >> 32);
  }

  template <typename Scalar, typename IndexType>
  void
  decompress_scatter_add_zfp_48_kernel(
    sycl::nd_item<1>  item,
    const uint16_t   *stream,
    const IndexType  *indices,
    unsigned int      gatherBlockSize,
    Scalar           *dataArray,
    unsigned int      tot_blocks)
  {
    const unsigned int block_idx = item.get_global_id(0);
    if (block_idx >= tot_blocks)
      return;

    size_t out    = (size_t)block_idx * 3u;
    uint64 packed = (uint64)stream[out] |
                    ((uint64)stream[out + 1] << 16) |
                    ((uint64)stream[out + 2] << 32);

    InlineBlockReader reader(packed);
    Scalar            fblock[4];
    decode_block<InlineBlockReader, Scalar>(reader, fblock, 48u);

    const unsigned int blocks_per_entry = gatherBlockSize >> 2;
    unsigned int       gatherIdx  = block_idx / blocks_per_entry;
    unsigned int       localBlock = block_idx - gatherIdx * blocks_per_entry;
    unsigned int       intraIdx   = localBlock * 4u;
    size_t base =
      (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

    portable_atomicAdd(&dataArray[base],     fblock[0]);
    portable_atomicAdd(&dataArray[base + 1], fblock[1]);
    portable_atomicAdd(&dataArray[base + 2], fblock[2]);
    portable_atomicAdd(&dataArray[base + 3], fblock[3]);
  }

#endif /* ZFP specialized bpt=1 kernels */

  /* =========================================================================
     Block Floating Point (BFP) compression

     Simpler and faster than ZFP-style bit-plane coding. Each block of 4
     values is packed into a single uint32_t: shared exponent + 4 uniform
     fixed-point quantized values. No lifting transform, no bit-plane loop,
     minimal branches.

     Layout per block (maxbits = bits_per_value * 4, LSB first):
       [0]                  continuation bit (0 = zero block)
       [1 .. EBITS]         biased exponent (EBITS bits)
       [EBITS+1 .. end]     4 x vbits-bit signed coefficients

     vbits = (maxbits - EBITS - 1) / 4

     Examples for double (EBITS=11, ebits=12):
       8 bpv: maxbits=32, vbits=5, total used=32  (exact fit)
       7 bpv: maxbits=28, vbits=4, total used=28  (exact fit)
       6 bpv: maxbits=24, vbits=3, total used=24  (exact fit)

     Examples for float (EBITS=8, ebits=9):
       8 bpv: maxbits=32, vbits=5, total used=29  (3 bits unused)
       7 bpv: maxbits=28, vbits=4, total used=25  (3 bits unused)
       6 bpv: maxbits=24, vbits=3, total used=21  (3 bits unused)

     Two kernel paths:
       bpv == 8:  uint32_t per block, 1 thread/block, 1 store (fastest)
       bpv != 8:  super-block packing via LocalBlockWriter (same buffer
                  layout and compressed_size as ZFP, but simpler encoding)

     Buffer size: compressed_size(num_values, bits_per_value) for all rates.
     =========================================================================
   */

  /* Device: BFP encode 4 values into a uint32_t */
  template <typename Scalar>
  COMPRESSION_DEVICE_INLINE unsigned int
  bfp_encode_block(Scalar *fblock, unsigned int maxbits)
  {
    const unsigned int ebits = (unsigned int)traits<Scalar>::EBITS + 1u;
    const unsigned int vbits = (maxbits - ebits) / 4u;

    int emax = max_exponent<Scalar>(fblock);
    int maxprec =
      calc_precision(emax, traits<Scalar>::PREC, traits<Scalar>::MINEXP);
    unsigned int e =
      maxprec ? (unsigned int)(emax + traits<Scalar>::EBIAS) : 0u;

    if (!e)
      return 0u; /* zero block */

    unsigned int packed = 2u * e + 1u; /* continuation + exponent */

    /* quantize: q_i = round_toward_zero(value_i * 2^(vbits-1) / 2^emax) */
    Scalar             s = portable_ldexp((Scalar)1.0, (int)vbits - 1 - emax);
    const unsigned int vmask = (1u << vbits) - 1u;
    for (int i = 0; i < 4; i++)
      {
        int q = (int)(s * fblock[i]);
        packed |= ((unsigned int)q & vmask)
                  << (ebits + (unsigned int)i * vbits);
      }
    return packed;
  }

  /* Device: BFP decode a uint32_t into 4 values */
  template <typename Scalar>
  COMPRESSION_DEVICE_INLINE void
  bfp_decode_block(unsigned int packed, Scalar *fblock, unsigned int maxbits)
  {
    const unsigned int ebits = (unsigned int)traits<Scalar>::EBITS + 1u;
    const unsigned int vbits = (maxbits - ebits) / 4u;

    if (!(packed & 1u))
      {
        for (int i = 0; i < 4; i++)
          fblock[i] = (Scalar)0;
        return;
      }

    const unsigned int vmask          = (1u << vbits) - 1u;
    const int          sign_threshold = 1 << (vbits - 1);

    unsigned int e_raw = (packed >> 1) & ((1u << (ebits - 1u)) - 1u);
    int          emax  = (int)e_raw - traits<Scalar>::EBIAS;

    /* dequantize: value_i = q_i * 2^(emax - vbits + 1) */
    Scalar scale = portable_ldexp((Scalar)1.0, emax - (int)vbits + 1);

    for (int i = 0; i < 4; i++)
      {
        unsigned int raw =
          (packed >> (ebits + (unsigned int)i * vbits)) & vmask;
        int q = (int)raw;
        if (q >= sign_threshold)
          q -= (1 << vbits); /* sign-extend */
        fblock[i] = scale * (Scalar)q;
      }
  }

  /* =========================================================================
     Specialized BFP 12-bpv (48-bit) encode/decode and packing helpers

     Eliminates LocalBlockWriter/Reader abstraction for 12 bpv. Each block
     encodes to exactly 48 bits stored in the lower bits of a uint64.
     4 blocks (192 bits) pack into exactly 3 × 64-bit Words via direct
     shift/OR — zero branches, zero division/modulo, zero conditional
     cross-word writes.

     Layout per block (48 bits, LSB first):
       float:  [0] cont + [1:8] exp(8) + [9:17] v0(9) + [18:26] v1(9)
               + [27:35] v2(9) + [36:44] v3(9) + [45:47] pad(3)
       double: [0] cont + [1:11] exp(11) + [12:20] v0(9) + [21:29] v1(9)
               + [30:38] v2(9) + [39:47] v3(9)

     Packing 4 × 48-bit blocks into 3 × 64-bit Words:
       word0 = b0[47:0]  | b1[15:0]<<48
       word1 = b1[47:16] | b2[31:0]<<32
       word2 = b2[47:32] | b3[47:0]<<16
     =========================================================================
   */

  template <typename Scalar>
  COMPRESSION_DEVICE_INLINE uint64
  bfp_encode_block_48(Scalar *fblock)
  {
    constexpr unsigned int ebits = (unsigned int)traits<Scalar>::EBITS + 1u;
    constexpr unsigned int vbits = (48u - ebits) / 4u;
    constexpr unsigned int vmask = (1u << vbits) - 1u;

    int emax = max_exponent<Scalar>(fblock);
    int maxprec =
      calc_precision(emax, traits<Scalar>::PREC, traits<Scalar>::MINEXP);
    unsigned int e =
      maxprec ? (unsigned int)(emax + traits<Scalar>::EBIAS) : 0u;

    if (!e)
      return (uint64)0;

    uint64 packed = (uint64)(2u * e + 1u);
    Scalar s      = portable_ldexp((Scalar)1.0, (int)vbits - 1 - emax);

    for (int i = 0; i < 4; i++)
      {
        int q = (int)(s * fblock[i]);
        packed |= ((uint64)((unsigned int)q & vmask))
                  << (ebits + (unsigned int)i * vbits);
      }
    return packed;
  }

  template <typename Scalar>
  COMPRESSION_DEVICE_INLINE void
  bfp_decode_block_48(uint64 packed, Scalar *fblock)
  {
    constexpr unsigned int ebits          = (unsigned int)traits<Scalar>::EBITS + 1u;
    constexpr unsigned int vbits          = (48u - ebits) / 4u;
    constexpr unsigned int vmask          = (1u << vbits) - 1u;
    constexpr int          sign_threshold = 1 << (vbits - 1);

    if (!(packed & 1ULL))
      {
        fblock[0] = fblock[1] = fblock[2] = fblock[3] = (Scalar)0;
        return;
      }

    unsigned int e_raw =
      (unsigned int)((packed >> 1) & ((1u << (ebits - 1u)) - 1u));
    int    emax  = (int)e_raw - traits<Scalar>::EBIAS;
    Scalar scale = portable_ldexp((Scalar)1.0, emax - (int)vbits + 1);

    for (int i = 0; i < 4; i++)
      {
        unsigned int raw = (unsigned int)(
          (packed >> (ebits + (unsigned int)i * vbits)) & vmask);
        int q = (int)raw;
        if (q >= sign_threshold)
          q -= (1 << vbits);
        fblock[i] = scale * (Scalar)q;
      }
  }

  /* Device: BFP encode via Writer (for super-block kernel, any bpv) */
  template <typename Writer, typename Scalar>
  COMPRESSION_DEVICE_INLINE void
  bfp_encode_block_writer(Writer &writer, Scalar *fblock, unsigned int maxbits)
  {
    const unsigned int ebits = (unsigned int)traits<Scalar>::EBITS + 1u;
    const unsigned int vbits = (maxbits - ebits) / 4u;

    int emax = max_exponent<Scalar>(fblock);
    int maxprec =
      calc_precision(emax, traits<Scalar>::PREC, traits<Scalar>::MINEXP);
    unsigned int e =
      maxprec ? (unsigned int)(emax + traits<Scalar>::EBIAS) : 0u;

    if (!e)
      return; /* zero block: write nothing (buffer is pre-zeroed) */

    writer.write_bits(2u * e + 1u, ebits); /* continuation + exponent */

    Scalar             s = portable_ldexp((Scalar)1.0, (int)vbits - 1 - emax);
    const unsigned int vmask = (1u << vbits) - 1u;
    for (int i = 0; i < 4; i++)
      {
        int q = (int)(s * fblock[i]);
        writer.write_bits((uint64)((unsigned int)q & vmask), vbits);
      }
  }

  /* Device: BFP decode via Reader (for decompress, any bpv) */
  template <typename Reader, typename Scalar>
  COMPRESSION_DEVICE_INLINE void
  bfp_decode_block_reader(Reader &reader, Scalar *fblock, unsigned int maxbits)
  {
    const unsigned int ebits = (unsigned int)traits<Scalar>::EBITS + 1u;
    const unsigned int vbits = (maxbits - ebits) / 4u;

    unsigned int s_cont = reader.read_bit();
    if (!s_cont)
      {
        for (int i = 0; i < 4; i++)
          fblock[i] = (Scalar)0;
        return;
      }

    int       emax  = (int)reader.read_bits(ebits - 1u) - traits<Scalar>::EBIAS;
    Scalar    scale = portable_ldexp((Scalar)1.0, emax - (int)vbits + 1);
    const int sign_threshold = 1 << (vbits - 1);

    for (int i = 0; i < 4; i++)
      {
        unsigned int raw = (unsigned int)reader.read_bits(vbits);
        int          q   = (int)raw;
        if (q >= sign_threshold)
          q -= (1 << vbits);
        fblock[i] = scale * (Scalar)q;
      }
  }

  /* --- BFP GPU kernels ---------------------------------------------------- */

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)

  template <typename Scalar>
  COMPRESSION_GLOBAL void
  compress_bfp_kernel(const Scalar *COMPRESSION_RESTRICT data,
                      unsigned int *COMPRESSION_RESTRICT stream,
                      unsigned int                       maxbits,
                      unsigned int                       dim,
                      unsigned int                       tot_blocks)
  {
    const unsigned int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= tot_blocks)
      return;

    unsigned int block_start = block_idx * 4u;
    Scalar       fblock[4];

    if (block_start + 4u <= dim)
      {
        fblock[0] = data[block_start];
        fblock[1] = data[block_start + 1];
        fblock[2] = data[block_start + 2];
        fblock[3] = data[block_start + 3];
      }
    else
      {
        unsigned int nx = dim - block_start;
        for (unsigned int i = 0; i < nx; i++)
          fblock[i] = data[block_start + i];
        pad_block(fblock, nx);
      }

    stream[block_idx] = bfp_encode_block(fblock, maxbits);
  }

  template <typename Scalar>
  COMPRESSION_GLOBAL void
  decompress_bfp_kernel(const unsigned int *COMPRESSION_RESTRICT stream,
                        Scalar *COMPRESSION_RESTRICT             data,
                        unsigned int                             maxbits,
                        unsigned int                             dim,
                        unsigned int                             tot_blocks)
  {
    const unsigned int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= tot_blocks)
      return;

    Scalar fblock[4];
    bfp_decode_block(stream[block_idx], fblock, maxbits);

    unsigned int block_start = block_idx * 4u;
    if (block_start + 4u <= dim)
      {
        data[block_start]     = fblock[0];
        data[block_start + 1] = fblock[1];
        data[block_start + 2] = fblock[2];
        data[block_start + 3] = fblock[3];
      }
    else
      {
        unsigned int nx = dim - block_start;
        for (unsigned int i = 0; i < nx; i++)
          data[block_start + i] = fblock[i];
      }
  }

#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)

  template <typename Scalar>
  void
  compress_bfp_kernel(sycl::nd_item<1> item,
                      const Scalar    *data,
                      unsigned int    *stream,
                      unsigned int     maxbits,
                      unsigned int     dim,
                      unsigned int     tot_blocks)
  {
    const unsigned int block_idx = item.get_global_id(0);
    if (block_idx >= tot_blocks)
      return;

    unsigned int block_start = block_idx * 4u;
    Scalar       fblock[4];

    if (block_start + 4u <= dim)
      {
        fblock[0] = data[block_start];
        fblock[1] = data[block_start + 1];
        fblock[2] = data[block_start + 2];
        fblock[3] = data[block_start + 3];
      }
    else
      {
        unsigned int nx = dim - block_start;
        for (unsigned int i = 0; i < nx; i++)
          fblock[i] = data[block_start + i];
        pad_block(fblock, nx);
      }

    stream[block_idx] = bfp_encode_block(fblock, maxbits);
  }

  template <typename Scalar>
  void
  decompress_bfp_kernel(sycl::nd_item<1>    item,
                        const unsigned int *stream,
                        Scalar             *data,
                        unsigned int        maxbits,
                        unsigned int        dim,
                        unsigned int        tot_blocks)
  {
    const unsigned int block_idx = item.get_global_id(0);
    if (block_idx >= tot_blocks)
      return;

    Scalar fblock[4];
    bfp_decode_block(stream[block_idx], fblock, maxbits);

    unsigned int block_start = block_idx * 4u;
    if (block_start + 4u <= dim)
      {
        data[block_start]     = fblock[0];
        data[block_start + 1] = fblock[1];
        data[block_start + 2] = fblock[2];
        data[block_start + 3] = fblock[3];
      }
    else
      {
        unsigned int nx = dim - block_start;
        for (unsigned int i = 0; i < nx; i++)
          data[block_start + i] = fblock[i];
      }
  }

#endif /* BFP uint32_t SYCL */

  /* --- Fused BFP gather+compress / decompress+scatter_add (uint32_t path) -- */

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)

  template <typename Scalar, typename IndexType>
  COMPRESSION_GLOBAL void
  compress_gather_bfp_kernel(const Scalar     *COMPRESSION_RESTRICT dataArray,
                             const IndexType  *COMPRESSION_RESTRICT indices,
                             unsigned int                           gatherBlockSize,
                             unsigned int *COMPRESSION_RESTRICT     stream,
                             unsigned int                           maxbits,
                             unsigned int                           tot_blocks)
  {
    const unsigned int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= tot_blocks)
      return;

    const unsigned int blocks_per_entry = gatherBlockSize >> 2;
    unsigned int       gatherIdx  = block_idx / blocks_per_entry;
    unsigned int       localBlock = block_idx - gatherIdx * blocks_per_entry;
    unsigned int       intraIdx   = localBlock * 4u;
    size_t base =
      (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

    Scalar fblock[4];
    fblock[0] = dataArray[base];
    fblock[1] = dataArray[base + 1];
    fblock[2] = dataArray[base + 2];
    fblock[3] = dataArray[base + 3];

    stream[block_idx] = bfp_encode_block(fblock, maxbits);
  }

  /* --- Fused BFP decompress+scatter_add kernel (uint32_t path) ------------ */

  template <typename Scalar, typename IndexType>
  COMPRESSION_GLOBAL void
  decompress_scatter_add_bfp_kernel(
    const unsigned int *COMPRESSION_RESTRICT stream,
    const IndexType    *COMPRESSION_RESTRICT indices,
    unsigned int                             gatherBlockSize,
    Scalar *COMPRESSION_RESTRICT             dataArray,
    unsigned int                             maxbits,
    unsigned int                             tot_blocks)
  {
    const unsigned int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= tot_blocks)
      return;

    Scalar fblock[4];
    bfp_decode_block(stream[block_idx], fblock, maxbits);

    const unsigned int blocks_per_entry = gatherBlockSize >> 2;
    unsigned int       gatherIdx  = block_idx / blocks_per_entry;
    unsigned int       localBlock = block_idx - gatherIdx * blocks_per_entry;
    unsigned int       intraIdx   = localBlock * 4u;
    size_t base =
      (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

    portable_atomicAdd(&dataArray[base],     fblock[0]);
    portable_atomicAdd(&dataArray[base + 1], fblock[1]);
    portable_atomicAdd(&dataArray[base + 2], fblock[2]);
    portable_atomicAdd(&dataArray[base + 3], fblock[3]);
  }

#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)

  template <typename Scalar, typename IndexType>
  void
  compress_gather_bfp_kernel(sycl::nd_item<1>   item,
                             const Scalar      *dataArray,
                             const IndexType   *indices,
                             unsigned int        gatherBlockSize,
                             unsigned int       *stream,
                             unsigned int        maxbits,
                             unsigned int        tot_blocks)
  {
    const unsigned int block_idx = item.get_global_id(0);
    if (block_idx >= tot_blocks)
      return;

    const unsigned int blocks_per_entry = gatherBlockSize >> 2;
    unsigned int       gatherIdx  = block_idx / blocks_per_entry;
    unsigned int       localBlock = block_idx - gatherIdx * blocks_per_entry;
    unsigned int       intraIdx   = localBlock * 4u;
    size_t base =
      (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

    Scalar fblock[4];
    fblock[0] = dataArray[base];
    fblock[1] = dataArray[base + 1];
    fblock[2] = dataArray[base + 2];
    fblock[3] = dataArray[base + 3];

    stream[block_idx] = bfp_encode_block(fblock, maxbits);
  }

  template <typename Scalar, typename IndexType>
  void
  decompress_scatter_add_bfp_kernel(
    sycl::nd_item<1>    item,
    const unsigned int *stream,
    const IndexType    *indices,
    unsigned int        gatherBlockSize,
    Scalar             *dataArray,
    unsigned int        maxbits,
    unsigned int        tot_blocks)
  {
    const unsigned int block_idx = item.get_global_id(0);
    if (block_idx >= tot_blocks)
      return;

    Scalar fblock[4];
    bfp_decode_block(stream[block_idx], fblock, maxbits);

    const unsigned int blocks_per_entry = gatherBlockSize >> 2;
    unsigned int       gatherIdx  = block_idx / blocks_per_entry;
    unsigned int       localBlock = block_idx - gatherIdx * blocks_per_entry;
    unsigned int       intraIdx   = localBlock * 4u;
    size_t base =
      (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

    portable_atomicAdd(&dataArray[base],     fblock[0]);
    portable_atomicAdd(&dataArray[base + 1], fblock[1]);
    portable_atomicAdd(&dataArray[base + 2], fblock[2]);
    portable_atomicAdd(&dataArray[base + 3], fblock[3]);
  }

#endif /* BFP uint32_t fused SYCL */

  /* =========================================================================
     Specialized BFP 12-bpv kernels (48-bit → 3 × uint16_t per block)

     1 thread per block, 3 × uint16_t stores. Mirrors the 8-bpv uint32_t
     design: each thread independently encodes one block, no cross-block
     packing. 4× more threads than the previous 4-block super-block design,
     better GPU saturation for small problem sizes (10^3-10^5 elements).

     Zero warp divergence. No atomics on compress path. Simple layout:
       block i → stream16[i*3], stream16[i*3+1], stream16[i*3+2]
     =========================================================================
   */

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)

  template <typename Scalar>
  COMPRESSION_GLOBAL void
  compress_bfp_12_kernel(const Scalar *COMPRESSION_RESTRICT data,
                         uint16_t *COMPRESSION_RESTRICT     stream,
                         unsigned int                       dim,
                         unsigned int                       tot_blocks)
  {
    const unsigned int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= tot_blocks)
      return;

    unsigned int block_start = block_idx * 4u;
    Scalar       fblock[4];

    if (block_start + 4u <= dim)
      {
        fblock[0] = data[block_start];
        fblock[1] = data[block_start + 1];
        fblock[2] = data[block_start + 2];
        fblock[3] = data[block_start + 3];
      }
    else
      {
        unsigned int nx = dim - block_start;
        for (unsigned int j = 0; j < nx; j++)
          fblock[j] = data[block_start + j];
        pad_block(fblock, nx);
      }

    uint64   packed = bfp_encode_block_48(fblock);
    size_t   out    = (size_t)block_idx * 3u;
    stream[out]     = (uint16_t)(packed);
    stream[out + 1] = (uint16_t)(packed >> 16);
    stream[out + 2] = (uint16_t)(packed >> 32);
  }

  template <typename Scalar>
  COMPRESSION_GLOBAL void
  decompress_bfp_12_kernel(const uint16_t *COMPRESSION_RESTRICT stream,
                           Scalar *COMPRESSION_RESTRICT         data,
                           unsigned int                         dim,
                           unsigned int                         tot_blocks)
  {
    const unsigned int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= tot_blocks)
      return;

    size_t out    = (size_t)block_idx * 3u;
    uint64 packed = (uint64)stream[out] |
                    ((uint64)stream[out + 1] << 16) |
                    ((uint64)stream[out + 2] << 32);

    Scalar fblock[4];
    bfp_decode_block_48(packed, fblock);

    unsigned int block_start = block_idx * 4u;
    if (block_start + 4u <= dim)
      {
        data[block_start]     = fblock[0];
        data[block_start + 1] = fblock[1];
        data[block_start + 2] = fblock[2];
        data[block_start + 3] = fblock[3];
      }
    else
      {
        unsigned int nx = dim - block_start;
        for (unsigned int j = 0; j < nx; j++)
          data[block_start + j] = fblock[j];
      }
  }

  template <typename Scalar, typename IndexType>
  COMPRESSION_GLOBAL void
  compress_gather_bfp_12_kernel(
    const Scalar    *COMPRESSION_RESTRICT dataArray,
    const IndexType *COMPRESSION_RESTRICT indices,
    unsigned int                          gatherBlockSize,
    uint16_t *COMPRESSION_RESTRICT        stream,
    unsigned int                          tot_blocks)
  {
    const unsigned int block_idx        = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= tot_blocks)
      return;

    const unsigned int blocks_per_entry = gatherBlockSize >> 2;
    unsigned int       gatherIdx        = block_idx / blocks_per_entry;
    unsigned int       localBlock       = block_idx - gatherIdx * blocks_per_entry;
    unsigned int       intraIdx         = localBlock * 4u;
    size_t             base =
      (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

    Scalar fblock[4];
    fblock[0] = dataArray[base];
    fblock[1] = dataArray[base + 1];
    fblock[2] = dataArray[base + 2];
    fblock[3] = dataArray[base + 3];

    uint64   packed = bfp_encode_block_48(fblock);
    size_t   out    = (size_t)block_idx * 3u;
    stream[out]     = (uint16_t)(packed);
    stream[out + 1] = (uint16_t)(packed >> 16);
    stream[out + 2] = (uint16_t)(packed >> 32);
  }

  template <typename Scalar, typename IndexType>
  COMPRESSION_GLOBAL void
  decompress_scatter_add_bfp_12_kernel(
    const uint16_t  *COMPRESSION_RESTRICT stream,
    const IndexType *COMPRESSION_RESTRICT indices,
    unsigned int                          gatherBlockSize,
    Scalar *COMPRESSION_RESTRICT          dataArray,
    unsigned int                          tot_blocks)
  {
    const unsigned int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= tot_blocks)
      return;

    size_t out    = (size_t)block_idx * 3u;
    uint64 packed = (uint64)stream[out] |
                    ((uint64)stream[out + 1] << 16) |
                    ((uint64)stream[out + 2] << 32);

    Scalar fblock[4];
    bfp_decode_block_48(packed, fblock);

    const unsigned int blocks_per_entry = gatherBlockSize >> 2;
    unsigned int       gatherIdx        = block_idx / blocks_per_entry;
    unsigned int       localBlock       = block_idx - gatherIdx * blocks_per_entry;
    unsigned int       intraIdx         = localBlock * 4u;
    size_t             base =
      (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

    portable_atomicAdd(&dataArray[base], fblock[0]);
    portable_atomicAdd(&dataArray[base + 1], fblock[1]);
    portable_atomicAdd(&dataArray[base + 2], fblock[2]);
    portable_atomicAdd(&dataArray[base + 3], fblock[3]);
  }

#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)

  template <typename Scalar>
  void
  compress_bfp_12_kernel(sycl::nd_item<1> item,
                         const Scalar    *data,
                         uint16_t        *stream,
                         unsigned int     dim,
                         unsigned int     tot_blocks)
  {
    const unsigned int block_idx = item.get_global_id(0);
    if (block_idx >= tot_blocks)
      return;

    unsigned int block_start = block_idx * 4u;
    Scalar       fblock[4];

    if (block_start + 4u <= dim)
      {
        fblock[0] = data[block_start];
        fblock[1] = data[block_start + 1];
        fblock[2] = data[block_start + 2];
        fblock[3] = data[block_start + 3];
      }
    else
      {
        unsigned int nx = dim - block_start;
        for (unsigned int j = 0; j < nx; j++)
          fblock[j] = data[block_start + j];
        pad_block(fblock, nx);
      }

    uint64   packed = bfp_encode_block_48(fblock);
    size_t   out    = (size_t)block_idx * 3u;
    stream[out]     = (uint16_t)(packed);
    stream[out + 1] = (uint16_t)(packed >> 16);
    stream[out + 2] = (uint16_t)(packed >> 32);
  }

  template <typename Scalar>
  void
  decompress_bfp_12_kernel(sycl::nd_item<1>  item,
                           const uint16_t   *stream,
                           Scalar           *data,
                           unsigned int      dim,
                           unsigned int      tot_blocks)
  {
    const unsigned int block_idx = item.get_global_id(0);
    if (block_idx >= tot_blocks)
      return;

    size_t out    = (size_t)block_idx * 3u;
    uint64 packed = (uint64)stream[out] |
                    ((uint64)stream[out + 1] << 16) |
                    ((uint64)stream[out + 2] << 32);

    Scalar fblock[4];
    bfp_decode_block_48(packed, fblock);

    unsigned int block_start = block_idx * 4u;
    if (block_start + 4u <= dim)
      {
        data[block_start]     = fblock[0];
        data[block_start + 1] = fblock[1];
        data[block_start + 2] = fblock[2];
        data[block_start + 3] = fblock[3];
      }
    else
      {
        unsigned int nx = dim - block_start;
        for (unsigned int j = 0; j < nx; j++)
          data[block_start + j] = fblock[j];
      }
  }

  template <typename Scalar, typename IndexType>
  void
  compress_gather_bfp_12_kernel(sycl::nd_item<1>  item,
                                const Scalar     *dataArray,
                                const IndexType  *indices,
                                unsigned int      gatherBlockSize,
                                uint16_t         *stream,
                                unsigned int      tot_blocks)
  {
    const unsigned int block_idx = item.get_global_id(0);
    if (block_idx >= tot_blocks)
      return;

    const unsigned int blocks_per_entry = gatherBlockSize >> 2;
    unsigned int       gatherIdx        = block_idx / blocks_per_entry;
    unsigned int       localBlock       = block_idx - gatherIdx * blocks_per_entry;
    unsigned int       intraIdx         = localBlock * 4u;
    size_t             base =
      (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

    Scalar fblock[4];
    fblock[0] = dataArray[base];
    fblock[1] = dataArray[base + 1];
    fblock[2] = dataArray[base + 2];
    fblock[3] = dataArray[base + 3];

    uint64   packed = bfp_encode_block_48(fblock);
    size_t   out    = (size_t)block_idx * 3u;
    stream[out]     = (uint16_t)(packed);
    stream[out + 1] = (uint16_t)(packed >> 16);
    stream[out + 2] = (uint16_t)(packed >> 32);
  }

  template <typename Scalar, typename IndexType>
  void
  decompress_scatter_add_bfp_12_kernel(sycl::nd_item<1>  item,
                                       const uint16_t   *stream,
                                       const IndexType  *indices,
                                       unsigned int      gatherBlockSize,
                                       Scalar           *dataArray,
                                       unsigned int      tot_blocks)
  {
    const unsigned int block_idx = item.get_global_id(0);
    if (block_idx >= tot_blocks)
      return;

    size_t out    = (size_t)block_idx * 3u;
    uint64 packed = (uint64)stream[out] |
                    ((uint64)stream[out + 1] << 16) |
                    ((uint64)stream[out + 2] << 32);

    Scalar fblock[4];
    bfp_decode_block_48(packed, fblock);

    const unsigned int blocks_per_entry = gatherBlockSize >> 2;
    unsigned int       gatherIdx        = block_idx / blocks_per_entry;
    unsigned int       localBlock       = block_idx - gatherIdx * blocks_per_entry;
    unsigned int       intraIdx         = localBlock * 4u;
    size_t             base =
      (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

    portable_atomicAdd(&dataArray[base], fblock[0]);
    portable_atomicAdd(&dataArray[base + 1], fblock[1]);
    portable_atomicAdd(&dataArray[base + 2], fblock[2]);
    portable_atomicAdd(&dataArray[base + 3], fblock[3]);
  }

#endif /* BFP 12bpv specialized SYCL */

  /* --- BFP super-block kernels (bpv > 8, tightly packed via Word stream) ---
   */

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)

  /* Super-block BFP decompress: each thread loads wpt Words into a local
     buffer, then decodes bpt blocks using LocalBlockReader (L1-cached reads).
     Mirrors compress_bfp_sb_kernel for bandwidth symmetry. */
  template <typename Scalar>
  COMPRESSION_GLOBAL void
  decompress_bfp_sb_kernel(const Word *COMPRESSION_RESTRICT stream,
                           Scalar *COMPRESSION_RESTRICT     data,
                           unsigned int                     maxbits,
                           unsigned int                     dim,
                           unsigned int                     tot_blocks,
                           unsigned int                     bpt,
                           unsigned int                     wpt,
                           unsigned int                     num_words)
  {
    const unsigned int super_idx  = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int block_base = super_idx * bpt;
    if (block_base >= tot_blocks)
      return;

    Word   local_words[MAX_WORDS_PER_SUPERBLOCK];
    size_t global_word_base = (size_t)super_idx * wpt;
    for (unsigned int w = 0; w < wpt; w++)
      {
        size_t gw      = global_word_base + w;
        local_words[w] = (gw < num_words) ? portable_ldg(stream + gw) : (Word)0;
      }

    for (unsigned int b = 0; b < bpt; b++)
      {
        unsigned int block_idx = block_base + b;
        if (block_idx >= tot_blocks)
          break;

        LocalBlockReader reader(local_words, maxbits, b);
        Scalar           fblock[4];
        bfp_decode_block_reader<LocalBlockReader, Scalar>(reader,
                                                          fblock,
                                                          maxbits);

        const unsigned int block_start = block_idx * 4u;
        if (block_start + 4u <= dim)
          {
            data[block_start]     = fblock[0];
            data[block_start + 1] = fblock[1];
            data[block_start + 2] = fblock[2];
            data[block_start + 3] = fblock[3];
          }
        else
          {
            unsigned int nx = dim - block_start;
            for (unsigned int i = 0; i < nx; i++)
              data[block_start + i] = fblock[i];
          }
      }
  }

  template <typename Scalar>
  COMPRESSION_GLOBAL void
  compress_bfp_sb_kernel(const Scalar *COMPRESSION_RESTRICT data,
                         Word *COMPRESSION_RESTRICT         stream,
                         unsigned int                       maxbits,
                         unsigned int                       dim,
                         unsigned int                       tot_blocks,
                         unsigned int                       bpt,
                         unsigned int                       wpt,
                         unsigned int                       num_words)
  {
    const unsigned int super_idx = blockIdx.x * blockDim.x + threadIdx.x;

    Word local_words[MAX_WORDS_PER_SUPERBLOCK];
    for (unsigned int w = 0; w < wpt; w++)
      local_words[w] = 0;

    const unsigned int block_base = super_idx * bpt;
    for (unsigned int b = 0; b < bpt; b++)
      {
        unsigned int block_idx = block_base + b;
        if (block_idx >= tot_blocks)
          break;

        unsigned int block_start = block_idx * 4u;
        Scalar       fblock[4];

        if (block_start + 4u <= dim)
          {
            fblock[0] = data[block_start];
            fblock[1] = data[block_start + 1];
            fblock[2] = data[block_start + 2];
            fblock[3] = data[block_start + 3];
          }
        else
          {
            unsigned int nx = dim - block_start;
            for (unsigned int i = 0; i < nx; i++)
              fblock[i] = data[block_start + i];
            pad_block(fblock, nx);
          }

        LocalBlockWriter writer(local_words, maxbits, b);
        bfp_encode_block_writer(writer, fblock, maxbits);
      }

    size_t global_word_base = (size_t)super_idx * wpt;
    for (unsigned int w = 0; w < wpt; w++)
      {
        size_t gw = global_word_base + w;
        if (gw < num_words)
          stream[gw] = local_words[w];
      }
  }

  /* --- Fused BFP gather+compress super-block kernel (CUDA/HIP, bpv > 8) --- */

  template <typename Scalar, typename IndexType>
  COMPRESSION_GLOBAL void
  compress_gather_bfp_sb_kernel(
    const Scalar     *COMPRESSION_RESTRICT dataArray,
    const IndexType  *COMPRESSION_RESTRICT indices,
    unsigned int                           gatherBlockSize,
    Word *COMPRESSION_RESTRICT             stream,
    unsigned int                           maxbits,
    unsigned int                           tot_blocks,
    unsigned int                           bpt,
    unsigned int                           wpt,
    unsigned int                           num_words)
  {
    const unsigned int super_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int blocks_per_entry = gatherBlockSize >> 2;

    Word local_words[MAX_WORDS_PER_SUPERBLOCK];
    for (unsigned int w = 0; w < wpt; w++)
      local_words[w] = 0;

    const unsigned int block_base = super_idx * bpt;
    for (unsigned int b = 0; b < bpt; b++)
      {
        unsigned int block_idx = block_base + b;
        if (block_idx >= tot_blocks)
          break;

        unsigned int gatherIdx  = block_idx / blocks_per_entry;
        unsigned int localBlock = block_idx - gatherIdx * blocks_per_entry;
        unsigned int intraIdx   = localBlock * 4u;
        size_t base =
          (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

        Scalar fblock[4];
        fblock[0] = dataArray[base];
        fblock[1] = dataArray[base + 1];
        fblock[2] = dataArray[base + 2];
        fblock[3] = dataArray[base + 3];

        LocalBlockWriter writer(local_words, maxbits, b);
        bfp_encode_block_writer(writer, fblock, maxbits);
      }

    size_t global_word_base = (size_t)super_idx * wpt;
    for (unsigned int w = 0; w < wpt; w++)
      {
        size_t gw = global_word_base + w;
        if (gw < num_words)
          stream[gw] = local_words[w];
      }
  }

  /* --- Fused BFP decompress+scatter_add super-block kernel (CUDA/HIP) ----- */

  template <typename Scalar, typename IndexType>
  COMPRESSION_GLOBAL void
  decompress_scatter_add_bfp_sb_kernel(
    const Word      *COMPRESSION_RESTRICT stream,
    const IndexType *COMPRESSION_RESTRICT indices,
    unsigned int                          gatherBlockSize,
    Scalar *COMPRESSION_RESTRICT          dataArray,
    unsigned int                          maxbits,
    unsigned int                          tot_blocks,
    unsigned int                          bpt,
    unsigned int                          wpt,
    unsigned int                          num_words)
  {
    const unsigned int super_idx  = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int block_base = super_idx * bpt;
    if (block_base >= tot_blocks)
      return;

    const unsigned int blocks_per_entry = gatherBlockSize >> 2;

    Word   local_words[MAX_WORDS_PER_SUPERBLOCK];
    size_t global_word_base = (size_t)super_idx * wpt;
    for (unsigned int w = 0; w < wpt; w++)
      {
        size_t gw      = global_word_base + w;
        local_words[w] = (gw < num_words) ? portable_ldg(stream + gw) : (Word)0;
      }

    for (unsigned int b = 0; b < bpt; b++)
      {
        unsigned int block_idx = block_base + b;
        if (block_idx >= tot_blocks)
          break;

        LocalBlockReader reader(local_words, maxbits, b);
        Scalar           fblock[4];
        bfp_decode_block_reader<LocalBlockReader, Scalar>(reader,
                                                          fblock,
                                                          maxbits);

        unsigned int gatherIdx  = block_idx / blocks_per_entry;
        unsigned int localBlock = block_idx - gatherIdx * blocks_per_entry;
        unsigned int intraIdx   = localBlock * 4u;
        size_t base =
          (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

        portable_atomicAdd(&dataArray[base],     fblock[0]);
        portable_atomicAdd(&dataArray[base + 1], fblock[1]);
        portable_atomicAdd(&dataArray[base + 2], fblock[2]);
        portable_atomicAdd(&dataArray[base + 3], fblock[3]);
      }
  }

#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)

  template <typename Scalar>
  void
  decompress_bfp_sb_kernel(sycl::nd_item<1> item,
                           const Word      *stream,
                           Scalar          *data,
                           unsigned int     maxbits,
                           unsigned int     dim,
                           unsigned int     tot_blocks,
                           unsigned int     bpt,
                           unsigned int     wpt,
                           unsigned int     num_words)
  {
    const unsigned int super_idx  = item.get_global_id(0);
    const unsigned int block_base = super_idx * bpt;
    if (block_base >= tot_blocks)
      return;

    Word   local_words[MAX_WORDS_PER_SUPERBLOCK];
    size_t global_word_base = (size_t)super_idx * wpt;
    for (unsigned int w = 0; w < wpt; w++)
      {
        size_t gw      = global_word_base + w;
        local_words[w] = (gw < num_words) ? stream[gw] : (Word)0;
      }

    for (unsigned int b = 0; b < bpt; b++)
      {
        unsigned int block_idx = block_base + b;
        if (block_idx >= tot_blocks)
          break;

        LocalBlockReader reader(local_words, maxbits, b);
        Scalar           fblock[4];
        bfp_decode_block_reader<LocalBlockReader, Scalar>(reader,
                                                          fblock,
                                                          maxbits);

        const unsigned int block_start = block_idx * 4u;
        if (block_start + 4u <= dim)
          {
            data[block_start]     = fblock[0];
            data[block_start + 1] = fblock[1];
            data[block_start + 2] = fblock[2];
            data[block_start + 3] = fblock[3];
          }
        else
          {
            unsigned int nx = dim - block_start;
            for (unsigned int i = 0; i < nx; i++)
              data[block_start + i] = fblock[i];
          }
      }
  }

  template <typename Scalar>
  void
  compress_bfp_sb_kernel(sycl::nd_item<1> item,
                         const Scalar    *data,
                         Word            *stream,
                         unsigned int     maxbits,
                         unsigned int     dim,
                         unsigned int     tot_blocks,
                         unsigned int     bpt,
                         unsigned int     wpt,
                         unsigned int     num_words)
  {
    const unsigned int super_idx = item.get_global_id(0);

    Word local_words[MAX_WORDS_PER_SUPERBLOCK];
    for (unsigned int w = 0; w < wpt; w++)
      local_words[w] = 0;

    const unsigned int block_base = super_idx * bpt;
    for (unsigned int b = 0; b < bpt; b++)
      {
        unsigned int block_idx = block_base + b;
        if (block_idx >= tot_blocks)
          break;

        unsigned int block_start = block_idx * 4u;
        Scalar       fblock[4];

        if (block_start + 4u <= dim)
          {
            fblock[0] = data[block_start];
            fblock[1] = data[block_start + 1];
            fblock[2] = data[block_start + 2];
            fblock[3] = data[block_start + 3];
          }
        else
          {
            unsigned int nx = dim - block_start;
            for (unsigned int i = 0; i < nx; i++)
              fblock[i] = data[block_start + i];
            pad_block(fblock, nx);
          }

        LocalBlockWriter writer(local_words, maxbits, b);
        bfp_encode_block_writer(writer, fblock, maxbits);
      }

    size_t global_word_base = (size_t)super_idx * wpt;
    for (unsigned int w = 0; w < wpt; w++)
      {
        size_t gw = global_word_base + w;
        if (gw < num_words)
          stream[gw] = local_words[w];
      }
  }

  /* --- Fused BFP gather+compress super-block kernel (SYCL, bpv > 8) ------- */

  template <typename Scalar, typename IndexType>
  void
  compress_gather_bfp_sb_kernel(
    sycl::nd_item<1>   item,
    const Scalar      *dataArray,
    const IndexType   *indices,
    unsigned int        gatherBlockSize,
    Word               *stream,
    unsigned int        maxbits,
    unsigned int        tot_blocks,
    unsigned int        bpt,
    unsigned int        wpt,
    unsigned int        num_words)
  {
    const unsigned int super_idx = item.get_global_id(0);
    const unsigned int blocks_per_entry = gatherBlockSize >> 2;

    Word local_words[MAX_WORDS_PER_SUPERBLOCK];
    for (unsigned int w = 0; w < wpt; w++)
      local_words[w] = 0;

    const unsigned int block_base = super_idx * bpt;
    for (unsigned int b = 0; b < bpt; b++)
      {
        unsigned int block_idx = block_base + b;
        if (block_idx >= tot_blocks)
          break;

        unsigned int gatherIdx  = block_idx / blocks_per_entry;
        unsigned int localBlock = block_idx - gatherIdx * blocks_per_entry;
        unsigned int intraIdx   = localBlock * 4u;
        size_t base =
          (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

        Scalar fblock[4];
        fblock[0] = dataArray[base];
        fblock[1] = dataArray[base + 1];
        fblock[2] = dataArray[base + 2];
        fblock[3] = dataArray[base + 3];

        LocalBlockWriter writer(local_words, maxbits, b);
        bfp_encode_block_writer(writer, fblock, maxbits);
      }

    size_t global_word_base = (size_t)super_idx * wpt;
    for (unsigned int w = 0; w < wpt; w++)
      {
        size_t gw = global_word_base + w;
        if (gw < num_words)
          stream[gw] = local_words[w];
      }
  }

  /* --- Fused BFP decompress+scatter_add super-block kernel (SYCL) --------- */

  template <typename Scalar, typename IndexType>
  void
  decompress_scatter_add_bfp_sb_kernel(
    sycl::nd_item<1>    item,
    const Word         *stream,
    const IndexType    *indices,
    unsigned int        gatherBlockSize,
    Scalar             *dataArray,
    unsigned int        maxbits,
    unsigned int        tot_blocks,
    unsigned int        bpt,
    unsigned int        wpt,
    unsigned int        num_words)
  {
    const unsigned int super_idx  = item.get_global_id(0);
    const unsigned int block_base = super_idx * bpt;
    if (block_base >= tot_blocks)
      return;

    const unsigned int blocks_per_entry = gatherBlockSize >> 2;

    Word   local_words[MAX_WORDS_PER_SUPERBLOCK];
    size_t global_word_base = (size_t)super_idx * wpt;
    for (unsigned int w = 0; w < wpt; w++)
      {
        size_t gw      = global_word_base + w;
        local_words[w] = (gw < num_words) ? stream[gw] : (Word)0;
      }

    for (unsigned int b = 0; b < bpt; b++)
      {
        unsigned int block_idx = block_base + b;
        if (block_idx >= tot_blocks)
          break;

        LocalBlockReader reader(local_words, maxbits, b);
        Scalar           fblock[4];
        bfp_decode_block_reader<LocalBlockReader, Scalar>(reader,
                                                          fblock,
                                                          maxbits);

        unsigned int gatherIdx  = block_idx / blocks_per_entry;
        unsigned int localBlock = block_idx - gatherIdx * blocks_per_entry;
        unsigned int intraIdx   = localBlock * 4u;
        size_t base =
          (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

        portable_atomicAdd(&dataArray[base],     fblock[0]);
        portable_atomicAdd(&dataArray[base + 1], fblock[1]);
        portable_atomicAdd(&dataArray[base + 2], fblock[2]);
        portable_atomicAdd(&dataArray[base + 3], fblock[3]);
      }
  }

#endif /* BFP super-block SYCL */

  /* =========================================================================
     Host API
     =========================================================================
   */

  /*
   * Compress a 1D GPU array using fixed-rate compression.
   *
   * Uses multi-block-per-thread direct writes: NO memset, NO atomics.
   * Each thread encodes bpt blocks (a "super-block") whose total bits
   * fill exactly wpt complete 64-bit Words, then writes them directly.
   */
  template <typename Scalar>
  inline void
  compress(const Scalar                *d_data,
           void                        *d_stream,
           size_t                       num_values,
           int                          bits_per_value,
           dftfe::utils::deviceStream_t stream = dftfe::utils::defaultStream)
  {
    if (num_values == 0)
      return;
    assert(num_values <= (size_t)UINT_MAX && "num_values exceeds 32-bit limit");
    assert(bits_per_value > 0 && "bits_per_value must be positive");

    const unsigned int dim        = (unsigned int)num_values;
    const unsigned int num_blocks = (dim + 3u) / 4u;
    const unsigned int maxbits    = (unsigned int)bits_per_value * 4u;

    if (maxbits == 32u)
      {
        /* --- bpt=1 fast path (8 bpv): 1 thread/block, 1 × uint32_t --- */
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        compress_zfp_32_kernel<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_data,
            reinterpret_cast<unsigned int *>(d_stream),
            dim,
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compress_zfp_32_kernel<Scalar>),
                           grid,
                           COMPRESSION_BLOCK_SIZE,
                           0,
                           stream,
                           d_data,
                           reinterpret_cast<unsigned int *>(d_stream),
                           dim,
                           num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto         &queue = dftfe::utils::queueRegistry.find(stream)->second;
        unsigned int *d_stream_u32 = reinterpret_cast<unsigned int *>(d_stream);
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             compress_zfp_32_kernel<Scalar>(item,
                                                            d_data,
                                                            d_stream_u32,
                                                            dim,
                                                            num_blocks);
                           });
#endif
      }
    else if (maxbits == 48u)
      {
        /* --- bpt=1 fast path (12 bpv): 1 thread/block, 3 × uint16_t --- */
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        compress_zfp_48_kernel<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_data,
            reinterpret_cast<uint16_t *>(d_stream),
            dim,
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compress_zfp_48_kernel<Scalar>),
                           grid,
                           COMPRESSION_BLOCK_SIZE,
                           0,
                           stream,
                           d_data,
                           reinterpret_cast<uint16_t *>(d_stream),
                           dim,
                           num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto     &queue      = dftfe::utils::queueRegistry.find(stream)->second;
        uint16_t *d_stream16 = reinterpret_cast<uint16_t *>(d_stream);
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             compress_zfp_48_kernel<Scalar>(item,
                                                            d_data,
                                                            d_stream16,
                                                            dim,
                                                            num_blocks);
                           });
#endif
      }
    else
      {
        /* --- generic super-block path for other bpv --- */
        unsigned int g   = compression_gcd(maxbits, (unsigned int)WSIZE);
        unsigned int bpt = (unsigned int)WSIZE / g;
        unsigned int wpt = maxbits / g;

        assert(wpt <= (unsigned int)MAX_WORDS_PER_SUPERBLOCK &&
               "bits_per_value requires too many words per super-block");

        unsigned int num_supers = (num_blocks + bpt - 1) / bpt;
        unsigned int num_words =
          (unsigned int)(compressed_size(num_values, bits_per_value) /
                         sizeof(Word));

        const unsigned int grid =
          (num_supers + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        compress_kernel<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_data,
            reinterpret_cast<Word *>(d_stream),
            maxbits,
            dim,
            num_blocks,
            bpt,
            wpt,
            num_words);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compress_kernel<Scalar>),
                           grid,
                           COMPRESSION_BLOCK_SIZE,
                           0,
                           stream,
                           d_data,
                           reinterpret_cast<Word *>(d_stream),
                           maxbits,
                           dim,
                           num_blocks,
                           bpt,
                           wpt,
                           num_words);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto &queue          = dftfe::utils::queueRegistry.find(stream)->second;
        Word *d_stream_words = reinterpret_cast<Word *>(d_stream);
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             compress_kernel<Scalar>(item,
                                                     d_data,
                                                     d_stream_words,
                                                     maxbits,
                                                     dim,
                                                     num_blocks,
                                                     bpt,
                                                     wpt,
                                                     num_words);
                           });
#endif
      }
  }

  /*
   * Decompress a 1D GPU array using fixed-rate compression.
   *
   * Uses the super-block kernel: each thread loads wpt Words into a
   * thread-local buffer (L1-cached) and decodes bpt blocks from it, mirroring
   * the compress super-block design for bandwidth symmetry and fewer cache
   * round-trips per decoded value.
   */
  template <typename Scalar>
  inline void
  decompress(const void                  *d_stream,
             Scalar                      *d_data,
             size_t                       num_values,
             int                          bits_per_value,
             dftfe::utils::deviceStream_t stream = dftfe::utils::defaultStream)
  {
    if (num_values == 0)
      return;
    assert(num_values <= (size_t)UINT_MAX && "num_values exceeds 32-bit limit");
    assert(bits_per_value > 0 && "bits_per_value must be positive");

    const unsigned int dim        = (unsigned int)num_values;
    const unsigned int num_blocks = (dim + 3u) / 4u;
    const unsigned int maxbits    = (unsigned int)bits_per_value * 4u;

    if (maxbits == 32u)
      {
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        decompress_zfp_32_kernel<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const unsigned int *>(d_stream),
            d_data,
            dim,
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(decompress_zfp_32_kernel<Scalar>),
                           grid,
                           COMPRESSION_BLOCK_SIZE,
                           0,
                           stream,
                           reinterpret_cast<const unsigned int *>(d_stream),
                           d_data,
                           dim,
                           num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto &queue = dftfe::utils::queueRegistry.find(stream)->second;
        const unsigned int *d_stream_u32 =
          reinterpret_cast<const unsigned int *>(d_stream);
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             decompress_zfp_32_kernel<Scalar>(item,
                                                               d_stream_u32,
                                                               d_data,
                                                               dim,
                                                               num_blocks);
                           });
#endif
      }
    else if (maxbits == 48u)
      {
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
        const uint16_t *d_stream16 =
          reinterpret_cast<const uint16_t *>(d_stream);
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        decompress_zfp_48_kernel<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_stream16, d_data, dim, num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(decompress_zfp_48_kernel<Scalar>),
                           grid,
                           COMPRESSION_BLOCK_SIZE,
                           0,
                           stream,
                           d_stream16,
                           d_data,
                           dim,
                           num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto &queue = dftfe::utils::queueRegistry.find(stream)->second;
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             decompress_zfp_48_kernel<Scalar>(item,
                                                               d_stream16,
                                                               d_data,
                                                               dim,
                                                               num_blocks);
                           });
#endif
      }
    else
      {
        unsigned int g   = compression_gcd(maxbits, (unsigned int)WSIZE);
        unsigned int bpt = (unsigned int)WSIZE / g;
        unsigned int wpt = maxbits / g;
        assert(wpt <= (unsigned int)MAX_WORDS_PER_SUPERBLOCK &&
               "bits_per_value requires too many words per super-block");

        unsigned int num_supers = (num_blocks + bpt - 1) / bpt;
        unsigned int num_words =
          (unsigned int)(compressed_size(num_values, bits_per_value) /
                         sizeof(Word));
        const unsigned int grid =
          (num_supers + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
        const Word *d_stream_words = reinterpret_cast<const Word *>(d_stream);

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        decompress_kernel_sb<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_stream_words, d_data, maxbits, dim, num_blocks,
            bpt, wpt, num_words);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(decompress_kernel_sb<Scalar>),
                           grid,
                           COMPRESSION_BLOCK_SIZE,
                           0,
                           stream,
                           d_stream_words,
                           d_data,
                           maxbits,
                           dim,
                           num_blocks,
                           bpt,
                           wpt,
                           num_words);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto &queue = dftfe::utils::queueRegistry.find(stream)->second;
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             decompress_kernel_sb<Scalar>(item,
                                                          d_stream_words,
                                                          d_data,
                                                          maxbits,
                                                          dim,
                                                          num_blocks,
                                                          bpt,
                                                          wpt,
                                                          num_words);
                           });
#endif
      }
  }

  /*
   * Fused gather+compress: reads scattered data via indirection and compresses
   * directly to the output stream. Eliminates the intermediate send buffer.
   *
   * @param dataArray       Full data array on device (scattered layout)
   * @param indices         Gather index array on device
   * @param num_indices     Number of index entries
   * @param gather_block_size Number of Scalar elements per index entry
   * @param d_stream        Output compressed buffer
   * @param bits_per_value  Compression rate
   * @param stream          Device stream
   */
  template <typename Scalar, typename IndexType>
  inline void
  compress_gather(
    const Scalar                *dataArray,
    const IndexType             *indices,
    size_t                       num_indices,
    unsigned int                 gather_block_size,
    void                        *d_stream,
    int                          bits_per_value,
    dftfe::utils::deviceStream_t stream = dftfe::utils::defaultStream)
  {
    const size_t num_values = num_indices * gather_block_size;
    if (num_values == 0)
      return;
    assert(num_values <= (size_t)UINT_MAX && "num_values exceeds 32-bit limit");
    assert(bits_per_value > 0 && "bits_per_value must be positive");

    const unsigned int dim        = (unsigned int)num_values;
    const unsigned int num_blocks = (dim + 3u) / 4u;
    const unsigned int maxbits    = (unsigned int)bits_per_value * 4u;

    if (maxbits == 32u)
      {
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        compress_gather_zfp_32_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            dataArray,
            indices,
            gather_block_size,
            reinterpret_cast<unsigned int *>(d_stream),
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(compress_gather_zfp_32_kernel<Scalar, IndexType>),
          grid,
          COMPRESSION_BLOCK_SIZE,
          0,
          stream,
          dataArray,
          indices,
          gather_block_size,
          reinterpret_cast<unsigned int *>(d_stream),
          num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto         &queue = dftfe::utils::queueRegistry.find(stream)->second;
        unsigned int *d_stream_u32 = reinterpret_cast<unsigned int *>(d_stream);
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             compress_gather_zfp_32_kernel<Scalar, IndexType>(
                               item,
                               dataArray,
                               indices,
                               gather_block_size,
                               d_stream_u32,
                               num_blocks);
                           });
#endif
      }
    else if (maxbits == 48u)
      {
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        compress_gather_zfp_48_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            dataArray,
            indices,
            gather_block_size,
            reinterpret_cast<uint16_t *>(d_stream),
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(compress_gather_zfp_48_kernel<Scalar, IndexType>),
          grid,
          COMPRESSION_BLOCK_SIZE,
          0,
          stream,
          dataArray,
          indices,
          gather_block_size,
          reinterpret_cast<uint16_t *>(d_stream),
          num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto     &queue      = dftfe::utils::queueRegistry.find(stream)->second;
        uint16_t *d_stream16 = reinterpret_cast<uint16_t *>(d_stream);
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             compress_gather_zfp_48_kernel<Scalar, IndexType>(
                               item,
                               dataArray,
                               indices,
                               gather_block_size,
                               d_stream16,
                               num_blocks);
                           });
#endif
      }
    else
      {
        unsigned int g   = compression_gcd(maxbits, (unsigned int)WSIZE);
        unsigned int bpt = (unsigned int)WSIZE / g;
        unsigned int wpt = maxbits / g;

        assert(wpt <= (unsigned int)MAX_WORDS_PER_SUPERBLOCK &&
               "bits_per_value requires too many words per super-block");

        unsigned int num_supers = (num_blocks + bpt - 1) / bpt;
        unsigned int num_words =
          (unsigned int)(compressed_size(num_values, bits_per_value) /
                         sizeof(Word));

        const unsigned int grid =
          (num_supers + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        compress_gather_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            dataArray,
            indices,
            gather_block_size,
            reinterpret_cast<Word *>(d_stream),
            maxbits,
            num_blocks,
            bpt,
            wpt,
            num_words);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(compress_gather_kernel<Scalar, IndexType>),
          grid,
          COMPRESSION_BLOCK_SIZE,
          0,
          stream,
          dataArray,
          indices,
          gather_block_size,
          reinterpret_cast<Word *>(d_stream),
          maxbits,
          num_blocks,
          bpt,
          wpt,
          num_words);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto &queue          = dftfe::utils::queueRegistry.find(stream)->second;
        Word *d_stream_words = reinterpret_cast<Word *>(d_stream);
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             compress_gather_kernel<Scalar, IndexType>(
                               item,
                               dataArray,
                               indices,
                               gather_block_size,
                               d_stream_words,
                               maxbits,
                               num_blocks,
                               bpt,
                               wpt,
                               num_words);
                           });
#endif
      }
  }

  /*
   * Fused decompress+scatter_add: decompresses and atomicAdds directly
   * to scattered positions in dataArray. Eliminates the intermediate buffer.
   *
   * @param d_stream        Input compressed buffer
   * @param indices         Scatter index array on device
   * @param num_indices     Number of index entries
   * @param gather_block_size Number of Scalar elements per index entry
   * @param dataArray       Full data array on device (accumulate target)
   * @param bits_per_value  Compression rate
   * @param stream          Device stream
   */
  template <typename Scalar, typename IndexType>
  inline void
  decompress_scatter_add(
    const void                  *d_stream,
    const IndexType             *indices,
    size_t                       num_indices,
    unsigned int                 gather_block_size,
    Scalar                      *dataArray,
    int                          bits_per_value,
    dftfe::utils::deviceStream_t stream = dftfe::utils::defaultStream)
  {
    const size_t num_values = num_indices * gather_block_size;
    if (num_values == 0)
      return;
    assert(num_values <= (size_t)UINT_MAX && "num_values exceeds 32-bit limit");
    assert(bits_per_value > 0 && "bits_per_value must be positive");

    const unsigned int dim        = (unsigned int)num_values;
    const unsigned int num_blocks = (dim + 3u) / 4u;
    const unsigned int maxbits    = (unsigned int)bits_per_value * 4u;

    if (maxbits == 32u)
      {
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
        const unsigned int *d_stream_u32 =
          reinterpret_cast<const unsigned int *>(d_stream);
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        decompress_scatter_add_zfp_32_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_stream_u32,
            indices,
            gather_block_size,
            dataArray,
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
            decompress_scatter_add_zfp_32_kernel<Scalar, IndexType>),
          grid,
          COMPRESSION_BLOCK_SIZE,
          0,
          stream,
          d_stream_u32,
          indices,
          gather_block_size,
          dataArray,
          num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto &queue = dftfe::utils::queueRegistry.find(stream)->second;
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             decompress_scatter_add_zfp_32_kernel<Scalar,
                                                                  IndexType>(
                               item,
                               d_stream_u32,
                               indices,
                               gather_block_size,
                               dataArray,
                               num_blocks);
                           });
#endif
      }
    else if (maxbits == 48u)
      {
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
        const uint16_t *d_stream16 =
          reinterpret_cast<const uint16_t *>(d_stream);
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        decompress_scatter_add_zfp_48_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_stream16,
            indices,
            gather_block_size,
            dataArray,
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
            decompress_scatter_add_zfp_48_kernel<Scalar, IndexType>),
          grid,
          COMPRESSION_BLOCK_SIZE,
          0,
          stream,
          d_stream16,
          indices,
          gather_block_size,
          dataArray,
          num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto &queue = dftfe::utils::queueRegistry.find(stream)->second;
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             decompress_scatter_add_zfp_48_kernel<Scalar,
                                                                  IndexType>(
                               item,
                               d_stream16,
                               indices,
                               gather_block_size,
                               dataArray,
                               num_blocks);
                           });
#endif
      }
    else
      {
        unsigned int g   = compression_gcd(maxbits, (unsigned int)WSIZE);
        unsigned int bpt = (unsigned int)WSIZE / g;
        unsigned int wpt = maxbits / g;
        assert(wpt <= (unsigned int)MAX_WORDS_PER_SUPERBLOCK &&
               "bits_per_value requires too many words per super-block");

        unsigned int num_supers = (num_blocks + bpt - 1) / bpt;
        unsigned int num_words =
          (unsigned int)(compressed_size(num_values, bits_per_value) /
                         sizeof(Word));
        const unsigned int grid =
          (num_supers + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
        const Word *d_stream_words = reinterpret_cast<const Word *>(d_stream);

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        decompress_scatter_add_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_stream_words,
            indices,
            gather_block_size,
            dataArray,
            maxbits,
            num_blocks,
            bpt,
            wpt,
            num_words);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
            decompress_scatter_add_kernel<Scalar, IndexType>),
          grid,
          COMPRESSION_BLOCK_SIZE,
          0,
          stream,
          d_stream_words,
          indices,
          gather_block_size,
          dataArray,
          maxbits,
          num_blocks,
          bpt,
          wpt,
          num_words);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto &queue = dftfe::utils::queueRegistry.find(stream)->second;
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             decompress_scatter_add_kernel<Scalar, IndexType>(
                               item,
                               d_stream_words,
                               indices,
                               gather_block_size,
                               dataArray,
                               maxbits,
                               num_blocks,
                               bpt,
                               wpt,
                               num_words);
                           });
#endif
      }
  }

  /*
   * BFP compress — block floating point for any bpv (4 to 16).
   *
   * No lifting transform, no bit-plane loop.
   *   bpv == 8:  uint32_t fast path (1 thread/block, 1 store)
   *   bpv != 8:  super-block path (same layout as ZFP, LocalBlockWriter)
   * Buffer: compressed_size(num_values, bits_per_value) bytes.
   */
  template <typename Scalar>
  inline void
  compress_bfp(
    const Scalar                *d_data,
    void                        *d_stream,
    size_t                       num_values,
    int                          bits_per_value,
    dftfe::utils::deviceStream_t stream = dftfe::utils::defaultStream)
  {
    if (num_values == 0)
      return;
    assert(num_values <= (size_t)UINT_MAX && "num_values exceeds 32-bit limit");
    assert(bits_per_value >= 4 && "BFP requires bits_per_value >= 4");

    const unsigned int dim        = (unsigned int)num_values;
    const unsigned int num_blocks = (dim + 3u) / 4u;
    const unsigned int maxbits    = (unsigned int)bits_per_value * 4u;
    const unsigned int ebits_chk  = (unsigned int)traits<Scalar>::EBITS + 1u;
    assert(
      maxbits >= ebits_chk + 4u &&
      "BFP requires bits_per_value large enough for exponent + at least 1 bit per value");

    if (maxbits == 32u)
      {
        /* --- fast uint32_t path: 1 thread/block, 1 store (bpv == 8) --- */
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        compress_bfp_kernel<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_data,
            reinterpret_cast<unsigned int *>(d_stream),
            maxbits,
            dim,
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compress_bfp_kernel<Scalar>),
                           grid,
                           COMPRESSION_BLOCK_SIZE,
                           0,
                           stream,
                           d_data,
                           reinterpret_cast<unsigned int *>(d_stream),
                           maxbits,
                           dim,
                           num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto         &queue = dftfe::utils::queueRegistry.find(stream)->second;
        unsigned int *d_stream_u32 = reinterpret_cast<unsigned int *>(d_stream);
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             compress_bfp_kernel<Scalar>(item,
                                                         d_data,
                                                         d_stream_u32,
                                                         maxbits,
                                                         dim,
                                                         num_blocks);
                           });
#endif
      }
    else if (maxbits == 48u)
      {
        /* --- specialized 48-bit path (bpv == 12): 1 thread/block, 3×uint16 --- */
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        compress_bfp_12_kernel<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_data,
            reinterpret_cast<uint16_t *>(d_stream),
            dim,
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compress_bfp_12_kernel<Scalar>),
                           grid,
                           COMPRESSION_BLOCK_SIZE,
                           0,
                           stream,
                           d_data,
                           reinterpret_cast<uint16_t *>(d_stream),
                           dim,
                           num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto     &queue      = dftfe::utils::queueRegistry.find(stream)->second;
        uint16_t *d_stream16 = reinterpret_cast<uint16_t *>(d_stream);
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             compress_bfp_12_kernel<Scalar>(item,
                                                            d_data,
                                                            d_stream16,
                                                            dim,
                                                            num_blocks);
                           });
#endif
      }
    else
      {
        /* --- generic super-block path for other bpv --- */
        unsigned int g   = compression_gcd(maxbits, (unsigned int)WSIZE);
        unsigned int bpt = (unsigned int)WSIZE / g;
        unsigned int wpt = maxbits / g;
        assert(wpt <= (unsigned int)MAX_WORDS_PER_SUPERBLOCK);

        unsigned int num_supers = (num_blocks + bpt - 1) / bpt;
        unsigned int num_words =
          (unsigned int)(compressed_size(num_values, bits_per_value) /
                         sizeof(Word));
        const unsigned int grid =
          (num_supers + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        compress_bfp_sb_kernel<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_data,
            reinterpret_cast<Word *>(d_stream),
            maxbits,
            dim,
            num_blocks,
            bpt,
            wpt,
            num_words);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compress_bfp_sb_kernel<Scalar>),
                           grid,
                           COMPRESSION_BLOCK_SIZE,
                           0,
                           stream,
                           d_data,
                           reinterpret_cast<Word *>(d_stream),
                           maxbits,
                           dim,
                           num_blocks,
                           bpt,
                           wpt,
                           num_words);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto &queue          = dftfe::utils::queueRegistry.find(stream)->second;
        Word *d_stream_words = reinterpret_cast<Word *>(d_stream);
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             compress_bfp_sb_kernel<Scalar>(item,
                                                            d_data,
                                                            d_stream_words,
                                                            maxbits,
                                                            dim,
                                                            num_blocks,
                                                            bpt,
                                                            wpt,
                                                            num_words);
                           });
#endif
      }
  }

  /*
   * BFP decompress — block floating point for any bpv (4 to 16).
   *
   *   bpv == 8:  uint32_t fast path
   *   bpv != 8:  BlockReader path (reads from packed Word stream)
   */
  template <typename Scalar>
  inline void
  decompress_bfp(
    const void                  *d_stream,
    Scalar                      *d_data,
    size_t                       num_values,
    int                          bits_per_value,
    dftfe::utils::deviceStream_t stream = dftfe::utils::defaultStream)
  {
    if (num_values == 0)
      return;
    assert(num_values <= (size_t)UINT_MAX && "num_values exceeds 32-bit limit");
    assert(bits_per_value >= 4 && "BFP requires bits_per_value >= 4");

    const unsigned int dim        = (unsigned int)num_values;
    const unsigned int num_blocks = (dim + 3u) / 4u;
    const unsigned int maxbits    = (unsigned int)bits_per_value * 4u;
    const unsigned int ebits_chk  = (unsigned int)traits<Scalar>::EBITS + 1u;
    assert(
      maxbits >= ebits_chk + 4u &&
      "BFP requires bits_per_value large enough for exponent + at least 1 bit per value");

    if (maxbits == 32u)
      {
        /* --- fast uint32_t path (bpv == 8): 1 thread per block --- */
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        decompress_bfp_kernel<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const unsigned int *>(d_stream),
            d_data,
            maxbits,
            dim,
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(decompress_bfp_kernel<Scalar>),
                           grid,
                           COMPRESSION_BLOCK_SIZE,
                           0,
                           stream,
                           reinterpret_cast<const unsigned int *>(d_stream),
                           d_data,
                           maxbits,
                           dim,
                           num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto &queue = dftfe::utils::queueRegistry.find(stream)->second;
        const unsigned int *d_stream_u32 =
          reinterpret_cast<const unsigned int *>(d_stream);
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             decompress_bfp_kernel<Scalar>(item,
                                                           d_stream_u32,
                                                           d_data,
                                                           maxbits,
                                                           dim,
                                                           num_blocks);
                           });
#endif
      }
    else if (maxbits == 48u)
      {
        /* --- specialized 48-bit path (bpv == 12): 1 thread/block, 3×uint16 --- */
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
        const uint16_t *d_stream16 =
          reinterpret_cast<const uint16_t *>(d_stream);
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        decompress_bfp_12_kernel<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_stream16, d_data, dim, num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(decompress_bfp_12_kernel<Scalar>),
                           grid,
                           COMPRESSION_BLOCK_SIZE,
                           0,
                           stream,
                           d_stream16,
                           d_data,
                           dim,
                           num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto &queue = dftfe::utils::queueRegistry.find(stream)->second;
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             decompress_bfp_12_kernel<Scalar>(item,
                                                              d_stream16,
                                                              d_data,
                                                              dim,
                                                              num_blocks);
                           });
#endif
      }
    else
      {
        /* --- generic super-block path for other bpv --- */
        unsigned int g   = compression_gcd(maxbits, (unsigned int)WSIZE);
        unsigned int bpt = (unsigned int)WSIZE / g;
        unsigned int wpt = maxbits / g;
        assert(wpt <= (unsigned int)MAX_WORDS_PER_SUPERBLOCK);

        unsigned int num_supers = (num_blocks + bpt - 1) / bpt;
        unsigned int num_words =
          (unsigned int)(compressed_size(num_values, bits_per_value) /
                         sizeof(Word));
        const unsigned int grid =
          (num_supers + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        decompress_bfp_sb_kernel<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const Word *>(d_stream),
            d_data,
            maxbits,
            dim,
            num_blocks,
            bpt,
            wpt,
            num_words);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(decompress_bfp_sb_kernel<Scalar>),
                           grid,
                           COMPRESSION_BLOCK_SIZE,
                           0,
                           stream,
                           reinterpret_cast<const Word *>(d_stream),
                           d_data,
                           maxbits,
                           dim,
                           num_blocks,
                           bpt,
                           wpt,
                           num_words);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto       &queue = dftfe::utils::queueRegistry.find(stream)->second;
        const Word *d_stream_words = reinterpret_cast<const Word *>(d_stream);
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             decompress_bfp_sb_kernel<Scalar>(item,
                                                              d_stream_words,
                                                              d_data,
                                                              maxbits,
                                                              dim,
                                                              num_blocks,
                                                              bpt,
                                                              wpt,
                                                              num_words);
                           });
#endif
      }
  }

  /*
   * Fused BFP gather+compress: reads scattered data via indices and compresses
   * directly. Supports both uint32_t fast path (bpv == 8) and super-block path.
   * Assumes gather_block_size is a multiple of 4.
   */
  template <typename Scalar, typename IndexType>
  inline void
  compress_gather_bfp(
    const Scalar                *dataArray,
    const IndexType             *indices,
    size_t                       num_indices,
    unsigned int                 gather_block_size,
    void                        *d_stream,
    int                          bits_per_value,
    dftfe::utils::deviceStream_t stream = dftfe::utils::defaultStream)
  {
    const size_t num_values = num_indices * gather_block_size;
    if (num_values == 0)
      return;
    assert(num_values <= (size_t)UINT_MAX && "num_values exceeds 32-bit limit");
    assert(bits_per_value >= 4 && "BFP requires bits_per_value >= 4");

    const unsigned int dim        = (unsigned int)num_values;
    const unsigned int num_blocks = (dim + 3u) / 4u;
    const unsigned int maxbits    = (unsigned int)bits_per_value * 4u;

    if (maxbits == 32u)
      {
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        compress_gather_bfp_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            dataArray,
            indices,
            gather_block_size,
            reinterpret_cast<unsigned int *>(d_stream),
            maxbits,
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(compress_gather_bfp_kernel<Scalar, IndexType>),
          grid,
          COMPRESSION_BLOCK_SIZE,
          0,
          stream,
          dataArray,
          indices,
          gather_block_size,
          reinterpret_cast<unsigned int *>(d_stream),
          maxbits,
          num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto         &queue = dftfe::utils::queueRegistry.find(stream)->second;
        unsigned int *d_stream_u32 = reinterpret_cast<unsigned int *>(d_stream);
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             compress_gather_bfp_kernel<Scalar, IndexType>(
                               item,
                               dataArray,
                               indices,
                               gather_block_size,
                               d_stream_u32,
                               maxbits,
                               num_blocks);
                           });
#endif
      }
    else if (maxbits == 48u)
      {
        /* --- specialized 48-bit path (bpv == 12): 1 thread/block, 3×uint16 --- */
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        compress_gather_bfp_12_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            dataArray,
            indices,
            gather_block_size,
            reinterpret_cast<uint16_t *>(d_stream),
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(compress_gather_bfp_12_kernel<Scalar, IndexType>),
          grid,
          COMPRESSION_BLOCK_SIZE,
          0,
          stream,
          dataArray,
          indices,
          gather_block_size,
          reinterpret_cast<uint16_t *>(d_stream),
          num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto     &queue      = dftfe::utils::queueRegistry.find(stream)->second;
        uint16_t *d_stream16 = reinterpret_cast<uint16_t *>(d_stream);
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             compress_gather_bfp_12_kernel<Scalar, IndexType>(
                               item,
                               dataArray,
                               indices,
                               gather_block_size,
                               d_stream16,
                               num_blocks);
                           });
#endif
      }
    else
      {
        unsigned int g   = compression_gcd(maxbits, (unsigned int)WSIZE);
        unsigned int bpt = (unsigned int)WSIZE / g;
        unsigned int wpt = maxbits / g;
        assert(wpt <= (unsigned int)MAX_WORDS_PER_SUPERBLOCK);

        unsigned int num_supers = (num_blocks + bpt - 1) / bpt;
        unsigned int num_words =
          (unsigned int)(compressed_size(num_values, bits_per_value) /
                         sizeof(Word));
        const unsigned int grid =
          (num_supers + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        compress_gather_bfp_sb_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            dataArray,
            indices,
            gather_block_size,
            reinterpret_cast<Word *>(d_stream),
            maxbits,
            num_blocks,
            bpt,
            wpt,
            num_words);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(compress_gather_bfp_sb_kernel<Scalar, IndexType>),
          grid,
          COMPRESSION_BLOCK_SIZE,
          0,
          stream,
          dataArray,
          indices,
          gather_block_size,
          reinterpret_cast<Word *>(d_stream),
          maxbits,
          num_blocks,
          bpt,
          wpt,
          num_words);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto &queue          = dftfe::utils::queueRegistry.find(stream)->second;
        Word *d_stream_words = reinterpret_cast<Word *>(d_stream);
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             compress_gather_bfp_sb_kernel<Scalar, IndexType>(
                               item,
                               dataArray,
                               indices,
                               gather_block_size,
                               d_stream_words,
                               maxbits,
                               num_blocks,
                               bpt,
                               wpt,
                               num_words);
                           });
#endif
      }
  }

  /*
   * Fused BFP decompress+scatter_add: decompresses and atomicAdds directly
   * to scattered positions in dataArray.
   * Assumes gather_block_size is a multiple of 4.
   */
  template <typename Scalar, typename IndexType>
  inline void
  decompress_scatter_add_bfp(
    const void                  *d_stream,
    const IndexType             *indices,
    size_t                       num_indices,
    unsigned int                 gather_block_size,
    Scalar                      *dataArray,
    int                          bits_per_value,
    dftfe::utils::deviceStream_t stream = dftfe::utils::defaultStream)
  {
    const size_t num_values = num_indices * gather_block_size;
    if (num_values == 0)
      return;
    assert(num_values <= (size_t)UINT_MAX && "num_values exceeds 32-bit limit");
    assert(bits_per_value >= 4 && "BFP requires bits_per_value >= 4");

    const unsigned int dim        = (unsigned int)num_values;
    const unsigned int num_blocks = (dim + 3u) / 4u;
    const unsigned int maxbits    = (unsigned int)bits_per_value * 4u;

    if (maxbits == 32u)
      {
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
        const unsigned int *d_stream_u32 =
          reinterpret_cast<const unsigned int *>(d_stream);
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        decompress_scatter_add_bfp_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_stream_u32,
            indices,
            gather_block_size,
            dataArray,
            maxbits,
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(decompress_scatter_add_bfp_kernel<Scalar, IndexType>),
          grid,
          COMPRESSION_BLOCK_SIZE,
          0,
          stream,
          d_stream_u32,
          indices,
          gather_block_size,
          dataArray,
          maxbits,
          num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto &queue = dftfe::utils::queueRegistry.find(stream)->second;
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             decompress_scatter_add_bfp_kernel<Scalar,
                                                               IndexType>(
                               item,
                               d_stream_u32,
                               indices,
                               gather_block_size,
                               dataArray,
                               maxbits,
                               num_blocks);
                           });
#endif
      }
    else if (maxbits == 48u)
      {
        /* --- specialized 48-bit path (bpv == 12): 1 thread/block, 3×uint16 --- */
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
        const uint16_t *d_stream16 =
          reinterpret_cast<const uint16_t *>(d_stream);
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        decompress_scatter_add_bfp_12_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_stream16,
            indices,
            gather_block_size,
            dataArray,
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
            decompress_scatter_add_bfp_12_kernel<Scalar, IndexType>),
          grid,
          COMPRESSION_BLOCK_SIZE,
          0,
          stream,
          d_stream16,
          indices,
          gather_block_size,
          dataArray,
          num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto &queue = dftfe::utils::queueRegistry.find(stream)->second;
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             decompress_scatter_add_bfp_12_kernel<Scalar,
                                                                  IndexType>(
                               item,
                               d_stream16,
                               indices,
                               gather_block_size,
                               dataArray,
                               num_blocks);
                           });
#endif
      }
    else
      {
        unsigned int g   = compression_gcd(maxbits, (unsigned int)WSIZE);
        unsigned int bpt = (unsigned int)WSIZE / g;
        unsigned int wpt = maxbits / g;
        assert(wpt <= (unsigned int)MAX_WORDS_PER_SUPERBLOCK);

        unsigned int num_supers = (num_blocks + bpt - 1) / bpt;
        unsigned int num_words =
          (unsigned int)(compressed_size(num_values, bits_per_value) /
                         sizeof(Word));
        const unsigned int grid =
          (num_supers + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
        const Word *d_stream_words = reinterpret_cast<const Word *>(d_stream);
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        decompress_scatter_add_bfp_sb_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_stream_words,
            indices,
            gather_block_size,
            dataArray,
            maxbits,
            num_blocks,
            bpt,
            wpt,
            num_words);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
            decompress_scatter_add_bfp_sb_kernel<Scalar, IndexType>),
          grid,
          COMPRESSION_BLOCK_SIZE,
          0,
          stream,
          d_stream_words,
          indices,
          gather_block_size,
          dataArray,
          maxbits,
          num_blocks,
          bpt,
          wpt,
          num_words);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto &queue = dftfe::utils::queueRegistry.find(stream)->second;
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             decompress_scatter_add_bfp_sb_kernel<Scalar,
                                                                  IndexType>(
                               item,
                               d_stream_words,
                               indices,
                               gather_block_size,
                               dataArray,
                               maxbits,
                               num_blocks,
                               bpt,
                               wpt,
                               num_words);
                           });
#endif
      }
  }

} /* namespace compression */

#endif /* COMPRESSION_H */
