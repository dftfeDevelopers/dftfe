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
**   - Compress kernel uses multi-block-per-thread direct writes:
**     no atomicAdd, no memset. Each thread encodes a "super-block"
**     of K blocks whose total bits fill complete 64-bit Words,
**     enabling coalesced direct stores.
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
     GPU kernels (portable across CUDA, HIP, SYCL)

     Super-block compress: each thread processes 'bpt' blocks whose combined
     output fills exactly 'wpt' complete 64-bit Words. Bits are accumulated
     in a thread-local buffer (LocalBlockWriter with |=), then flushed to
     global memory with direct stores. No atomics, no memset.

     Parameters computed on host:
       g   = gcd(maxbits, 64)
       bpt = 64 / g          (blocks per thread)
       wpt = maxbits / g     (words per thread)
     Guarantee: bpt * maxbits = wpt * 64 (word-aligned)

     Examples:
       8 bpv:  maxbits=32,  bpt=2,  wpt=1  (2 blocks → 1 word)
       12 bpv: maxbits=48,  bpt=4,  wpt=3  (4 blocks → 3 words)
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

#endif /* DFTFE_WITH_DEVICE_LANG_SYCL */

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

    /* super-block parameters */
    unsigned int g   = compression_gcd(maxbits, (unsigned int)WSIZE);
    unsigned int bpt = (unsigned int)WSIZE / g; /* blocks per thread */
    unsigned int wpt = maxbits / g;             /* words per thread  */

    assert(wpt <= (unsigned int)MAX_WORDS_PER_SUPERBLOCK &&
           "bits_per_value requires too many words per super-block");

    unsigned int num_supers = (num_blocks + bpt - 1) / bpt;
    unsigned int num_words =
      (unsigned int)(compressed_size(num_values, bits_per_value) /
                     sizeof(Word));

    const unsigned int grid =
      (num_supers + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;

    /* NO memset — local buffers are zero-initialized per thread */

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
    compress_kernel<Scalar>
      <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(d_data,
                                                    reinterpret_cast<Word *>(
                                                      d_stream),
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

    /* super-block parameters (same as compress) */
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
    decompress_kernel_sb<Scalar><<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
      d_stream_words, d_data, maxbits, dim, num_blocks, bpt, wpt, num_words);
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
        /* --- fast uint32_t path: 1 thread per block, 1 store (bpv == 8) --- */
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
    else
      {
        /* --- super-block path for bpv > 8: tightly packed Word stream --- */
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
    else
      {
        /* --- super-block path for bpv > 8 (mirrors compress_bfp_sb_kernel) ---
         */
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

} /* namespace compression */

#endif /* COMPRESSION_H */
