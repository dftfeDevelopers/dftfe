/*
** compressionKernels.h - GPU compressor: Block Floating Point (BFP)
**                       encode/decode helpers and fixed-rate kernels.
**
**   Core device helpers (constexpr-vbits fast paths):
**     encode_block_32 / decode_block_32   – uint32_t  (8 bpv)
**     encode_block_40 / decode_block_40   – uint64 lower 40 bits (10 bpv)
**     encode_block_48 / decode_block_48   – uint64 lower 48 bits (12 bpv)
**     encode_block_64 / decode_block_64   – uint64    (16 bpv)
**
**   GPU kernels (1 thread per 4-value block):
**     8  bpv (uint32_t):    compress_8_kernel, decompress_8_kernel,
**                           compress_gather_8_kernel, decompress_scatter_add_8_kernel
**     10 bpv (5×uint8_t):   compress_10_kernel, decompress_10_kernel,
**                           compress_gather_10_kernel, decompress_scatter_add_10_kernel
**     12 bpv (3×uint16_t):  compress_12_kernel, decompress_12_kernel,
**                           compress_gather_12_kernel, decompress_scatter_add_12_kernel
**     16 bpv (uint64):      compress_16_kernel, decompress_16_kernel,
**                           compress_gather_16_kernel, decompress_scatter_add_16_kernel
**
** CUDA/HIP and SYCL backends are both included via preprocessor guards.
** Included by compression.h. Never included directly by application code.
*/

#ifndef COMPRESSION_KERNELS_H
#define COMPRESSION_KERNELS_H
#include <compressionTypes.h>
namespace compression
{

  /* =========================================================================
     Block Floating Point (BFP) compression

     Each block of 4 values is packed into a fixed-width word: shared
     exponent + 4 uniform fixed-point quantized values. No lifting
     transform, no bit-plane loop, minimal branches.

     Layout per block (maxbits = bits_per_value * 4, LSB first):
       [0 .. EBITS-1]       biased exponent (EBITS bits, 0 = zero block)
       [EBITS .. end]       4 x vbits-bit signed coefficients
     vbits = (maxbits - EBITS) / 4
     =========================================================================
   */

  /* =========================================================================
     Specialized BFP 12-bpv (48-bit) encode/decode helpers

     Specialized helper for12 bpv. Each block
     encodes to exactly 48 bits returned in the lower bits of a uint64.
     The bpt=1 kernel (compress_12_kernel) stores these as 3 × uint16_t
     per block — 1 thread per block, 3 contiguous stores, no cross-block
     packing.

     Layout per block (48 bits, LSB first):
       float:  [0:7] exp(8) + [8:17] v0(10) + [18:27] v1(10)
               + [28:37] v2(10) + [38:47] v3(10)
       double: [0:10] exp(11) + [11:19] v0(9) + [20:28] v1(9)
               + [29:37] v2(9) + [38:46] v3(9) + [47] unused(1)

     vbits: float = (48 - 8) / 4 = 10, double = (48 - 11) / 4 = 9.
     float is an exact fit; double has 1 unused bit at [47].
     =========================================================================
   */

  template <typename Scalar>
  COMPRESSION_DEVICE_INLINE uint64
  encode_block_48(Scalar *fblock)
  {
    constexpr unsigned int ebits = (unsigned int)traits<Scalar>::EBITS;
    constexpr unsigned int vbits = (48u - ebits) / 4u;
    constexpr unsigned int vmask = (1u << vbits) - 1u;

    int emax = max_exponent<Scalar>(fblock);
    int maxprec =
      calc_precision(emax, traits<Scalar>::PREC, traits<Scalar>::MINEXP);
    unsigned int e =
      maxprec ? (unsigned int)(emax + traits<Scalar>::EBIAS) : 0u;

    if (!e)
      return (uint64)0;

    uint64    packed = (uint64)e;
    Scalar    s      = portable_ldexp((Scalar)1.0, (int)vbits - 1 - emax);
    const int qmax   = (int)(vmask >> 1u);

    for (int i = 0; i < 4; i++)
      {
        const int q_raw = (int)rint(s * fblock[i]);
        const int q     = q_raw > qmax ? qmax : q_raw;
        packed |= ((uint64)((unsigned int)q & vmask))
                  << (ebits + (unsigned int)i * vbits);
      }
    return packed;
  }

  template <typename Scalar>
  COMPRESSION_DEVICE_INLINE void
  decode_block_48(uint64 packed, Scalar *fblock)
  {
    constexpr unsigned int ebits          = (unsigned int)traits<Scalar>::EBITS;
    constexpr unsigned int vbits          = (48u - ebits) / 4u;
    constexpr unsigned int vmask          = (1u << vbits) - 1u;
    constexpr int          sign_threshold = 1 << (vbits - 1);
    constexpr unsigned int emask          = (1u << ebits) - 1u;

    unsigned int e_raw = (unsigned int)(packed & emask);
    if (!e_raw)
      {
        fblock[0] = fblock[1] = fblock[2] = fblock[3] = (Scalar)0;
        return;
      }

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

  /* =========================================================================
     Specialized BFP 10-bpv (40-bit) encode/decode helpers

     Specialized helper for10 bpv. Each block
     encodes to exactly 40 bits returned in the lower bits of a uint64.
     The bpt=1 kernel (compress_10_kernel) stores these as 5 × uint8_t
     per block — 1 thread per block, 5 contiguous stores, no cross-block
     packing.

     Layout per block (40 bits, LSB first):
       float:  [0:7]  exp(8)  + [8:15]  v0(8) + [16:23] v1(8)
               + [24:31] v2(8) + [32:39] v3(8)
       double: [0:10] exp(11) + [11:17] v0(7) + [18:24] v1(7)
               + [25:31] v2(7) + [32:38] v3(7) + [39] unused(1)

     vbits: float = (40 - 8) / 4 = 8, double = (40 - 11) / 4 = 7.
     float is an exact fit; double has 1 unused bit at [39].
     =========================================================================
   */

  template <typename Scalar>
  COMPRESSION_DEVICE_INLINE uint64
  encode_block_40(Scalar *fblock)
  {
    constexpr unsigned int ebits = (unsigned int)traits<Scalar>::EBITS;
    constexpr unsigned int vbits = (40u - ebits) / 4u;
    constexpr unsigned int vmask = (1u << vbits) - 1u;

    int emax = max_exponent<Scalar>(fblock);
    int maxprec =
      calc_precision(emax, traits<Scalar>::PREC, traits<Scalar>::MINEXP);
    unsigned int e =
      maxprec ? (unsigned int)(emax + traits<Scalar>::EBIAS) : 0u;

    if (!e)
      return (uint64)0;

    uint64    packed = (uint64)e;
    Scalar    s      = portable_ldexp((Scalar)1.0, (int)vbits - 1 - emax);
    const int qmax   = (int)(vmask >> 1u);

    for (int i = 0; i < 4; i++)
      {
        const int q_raw = (int)rint(s * fblock[i]);
        const int q     = q_raw > qmax ? qmax : q_raw;
        packed |= ((uint64)((unsigned int)q & vmask))
                  << (ebits + (unsigned int)i * vbits);
      }
    return packed;
  }

  template <typename Scalar>
  COMPRESSION_DEVICE_INLINE void
  decode_block_40(uint64 packed, Scalar *fblock)
  {
    constexpr unsigned int ebits          = (unsigned int)traits<Scalar>::EBITS;
    constexpr unsigned int vbits          = (40u - ebits) / 4u;
    constexpr unsigned int vmask          = (1u << vbits) - 1u;
    constexpr int          sign_threshold = 1 << (vbits - 1);
    constexpr unsigned int emask          = (1u << ebits) - 1u;

    unsigned int e_raw = (unsigned int)(packed & emask);
    if (!e_raw)
      {
        fblock[0] = fblock[1] = fblock[2] = fblock[3] = (Scalar)0;
        return;
      }

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

  /* =========================================================================
     Specialized BFP 8-bpv (32-bit) encode/decode

     Eliminates runtime vbits/vmask computation for the 8-bpv path. All
     constants are constexpr. Returns unsigned int (exactly 32 bits),
     compatible with the uint32_t stream layout.

     Layout per block (32 bits, LSB first):
       float:  [0:7]  exp(8)  + [8:13]  v0(6) + [14:19] v1(6)
               + [20:25] v2(6) + [26:31] v3(6)
       double: [0:10] exp(11) + [11:15] v0(5) + [16:20] v1(5)
               + [21:25] v2(5) + [26:30] v3(5) + [31] pad(1)
     =========================================================================
   */

  template <typename Scalar>
  COMPRESSION_DEVICE_INLINE unsigned int
  encode_block_32(Scalar *fblock)
  {
    constexpr unsigned int ebits = (unsigned int)traits<Scalar>::EBITS;
    constexpr unsigned int vbits = (32u - ebits) / 4u;
    constexpr unsigned int vmask = (1u << vbits) - 1u;

    int emax = max_exponent<Scalar>(fblock);
    int maxprec =
      calc_precision(emax, traits<Scalar>::PREC, traits<Scalar>::MINEXP);
    unsigned int e =
      maxprec ? (unsigned int)(emax + traits<Scalar>::EBIAS) : 0u;

    if (!e)
      return 0u;

    unsigned int packed = e;
    Scalar       s      = portable_ldexp((Scalar)1.0, (int)vbits - 1 - emax);
    const int    qmax   = (int)(vmask >> 1u);

    for (int i = 0; i < 4; i++)
      {
        const int q_raw = (int)rint(s * fblock[i]);
        const int q     = q_raw > qmax ? qmax : q_raw;
        packed |= ((unsigned int)q & vmask)
                  << (ebits + (unsigned int)i * vbits);
      }
    return packed;
  }

  template <typename Scalar>
  COMPRESSION_DEVICE_INLINE void
  decode_block_32(unsigned int packed, Scalar *fblock)
  {
    constexpr unsigned int ebits          = (unsigned int)traits<Scalar>::EBITS;
    constexpr unsigned int vbits          = (32u - ebits) / 4u;
    constexpr unsigned int vmask          = (1u << vbits) - 1u;
    constexpr int          sign_threshold = 1 << (vbits - 1);
    constexpr unsigned int emask          = (1u << ebits) - 1u;

    unsigned int e_raw = packed & emask;
    if (!e_raw)
      {
        fblock[0] = fblock[1] = fblock[2] = fblock[3] = (Scalar)0;
        return;
      }

    int    emax  = (int)e_raw - traits<Scalar>::EBIAS;
    Scalar scale = portable_ldexp((Scalar)1.0, emax - (int)vbits + 1);

    for (int i = 0; i < 4; i++)
      {
        unsigned int raw =
          (packed >> (ebits + (unsigned int)i * vbits)) & vmask;
        int q = (int)raw;
        if (q >= sign_threshold)
          q -= (1 << vbits);
        fblock[i] = scale * (Scalar)q;
      }
  }

  /* =========================================================================
     Specialized BFP 16-bpv (64-bit) encode/decode

     Specialized helper for16 bpv. Each block
     encodes to exactly 64 bits — a single uint64 Word. 1 thread per block,
     1 × uint64 store. All bit widths are compile-time constants.

     Layout per block (64 bits, LSB first):
       float:  [0:7]  exp(8)  + [8:21]  v0(14) + [22:35] v1(14)
               + [36:49] v2(14) + [50:63] v3(14)
       double: [0:10] exp(11) + [11:23] v0(13) + [24:36] v1(13)
               + [37:49] v2(13) + [50:62] v3(13) + [63] pad(1)
     =========================================================================
   */

  template <typename Scalar>
  COMPRESSION_DEVICE_INLINE uint64
  encode_block_64(Scalar *fblock)
  {
    constexpr unsigned int ebits = (unsigned int)traits<Scalar>::EBITS;
    constexpr unsigned int vbits = (64u - ebits) / 4u;
    constexpr unsigned int vmask = (1u << vbits) - 1u;

    int emax = max_exponent<Scalar>(fblock);
    int maxprec =
      calc_precision(emax, traits<Scalar>::PREC, traits<Scalar>::MINEXP);
    unsigned int e =
      maxprec ? (unsigned int)(emax + traits<Scalar>::EBIAS) : 0u;

    if (!e)
      return (uint64)0;

    uint64    packed = (uint64)e;
    Scalar    s      = portable_ldexp((Scalar)1.0, (int)vbits - 1 - emax);
    const int qmax   = (int)(vmask >> 1u);

    for (int i = 0; i < 4; i++)
      {
        const int q_raw = (int)rint(s * fblock[i]);
        const int q     = q_raw > qmax ? qmax : q_raw;
        packed |= ((uint64)((unsigned int)q & vmask))
                  << (ebits + (unsigned int)i * vbits);
      }
    return packed;
  }

  template <typename Scalar>
  COMPRESSION_DEVICE_INLINE void
  decode_block_64(uint64 packed, Scalar *fblock)
  {
    constexpr unsigned int ebits          = (unsigned int)traits<Scalar>::EBITS;
    constexpr unsigned int vbits          = (64u - ebits) / 4u;
    constexpr unsigned int vmask          = (1u << vbits) - 1u;
    constexpr int          sign_threshold = 1 << (vbits - 1);
    constexpr unsigned int emask          = (1u << ebits) - 1u;

    unsigned int e_raw = (unsigned int)(packed & emask);
    if (!e_raw)
      {
        fblock[0] = fblock[1] = fblock[2] = fblock[3] = (Scalar)0;
        return;
      }

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

  /* =========================================================================
     Specialized BFP 12-bpv kernels (48-bit → 3 × uint16_t per block)

     1 thread per block, 3 × uint16_t stores. Each thread independently
     encodes one block, no cross-block packing.

     Zero warp divergence. No atomics on compress path. Simple layout:
       block i → stream16[i*3], stream16[i*3+1], stream16[i*3+2]
     =========================================================================
   */

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)

  template <typename Scalar>
  COMPRESSION_GLOBAL void
  compress_12_kernel(const Scalar *COMPRESSION_RESTRICT data,
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

    uint64   packed = encode_block_48(fblock);
    size_t   out    = (size_t)block_idx * 3u;
    stream[out]     = (uint16_t)(packed);
    stream[out + 1] = (uint16_t)(packed >> 16);
    stream[out + 2] = (uint16_t)(packed >> 32);
  }

  template <typename Scalar>
  COMPRESSION_GLOBAL void
  decompress_12_kernel(const uint16_t *COMPRESSION_RESTRICT stream,
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
    decode_block_48(packed, fblock);

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
  compress_gather_12_kernel(
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

    uint64   packed = encode_block_48(fblock);
    size_t   out    = (size_t)block_idx * 3u;
    stream[out]     = (uint16_t)(packed);
    stream[out + 1] = (uint16_t)(packed >> 16);
    stream[out + 2] = (uint16_t)(packed >> 32);
  }

  template <typename Scalar, typename IndexType>
  COMPRESSION_GLOBAL void
  decompress_scatter_add_12_kernel(
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
    decode_block_48(packed, fblock);

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
  compress_12_kernel(sycl::nd_item<1> item,
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

    uint64   packed = encode_block_48(fblock);
    size_t   out    = (size_t)block_idx * 3u;
    stream[out]     = (uint16_t)(packed);
    stream[out + 1] = (uint16_t)(packed >> 16);
    stream[out + 2] = (uint16_t)(packed >> 32);
  }

  template <typename Scalar>
  void
  decompress_12_kernel(sycl::nd_item<1>  item,
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
    decode_block_48(packed, fblock);

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
  compress_gather_12_kernel(sycl::nd_item<1>  item,
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

    uint64   packed = encode_block_48(fblock);
    size_t   out    = (size_t)block_idx * 3u;
    stream[out]     = (uint16_t)(packed);
    stream[out + 1] = (uint16_t)(packed >> 16);
    stream[out + 2] = (uint16_t)(packed >> 32);
  }

  template <typename Scalar, typename IndexType>
  void
  decompress_scatter_add_12_kernel(sycl::nd_item<1>  item,
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
    decode_block_48(packed, fblock);

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

#endif /* BFP 12bpv specialized */

  /* =========================================================================
     Specialized BFP 10-bpv kernels (40-bit → 5 × uint8_t per block)

     1 thread per block, 5 × uint8_t stores. Mirrors the 12-bpv 3×uint16_t
     design. Each thread independently encodes one block, no cross-block
     packing.

     No atomics on compress path. Simple layout:
       block i → stream8[i*5], stream8[i*5+1], ..., stream8[i*5+4]
     =========================================================================
   */

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)

  template <typename Scalar>
  COMPRESSION_GLOBAL void
  compress_10_kernel(const Scalar *COMPRESSION_RESTRICT data,
                         uint8_t *COMPRESSION_RESTRICT      stream,
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

    uint64   packed = encode_block_40(fblock);
    size_t   out    = (size_t)block_idx * 5u;
    stream[out]     = (uint8_t)(packed);
    stream[out + 1] = (uint8_t)(packed >> 8);
    stream[out + 2] = (uint8_t)(packed >> 16);
    stream[out + 3] = (uint8_t)(packed >> 24);
    stream[out + 4] = (uint8_t)(packed >> 32);
  }

  template <typename Scalar>
  COMPRESSION_GLOBAL void
  decompress_10_kernel(const uint8_t *COMPRESSION_RESTRICT stream,
                           Scalar *COMPRESSION_RESTRICT        data,
                           unsigned int                        dim,
                           unsigned int                        tot_blocks)
  {
    const unsigned int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= tot_blocks)
      return;

    size_t out    = (size_t)block_idx * 5u;
    uint64 packed = (uint64)stream[out] |
                    ((uint64)stream[out + 1] << 8) |
                    ((uint64)stream[out + 2] << 16) |
                    ((uint64)stream[out + 3] << 24) |
                    ((uint64)stream[out + 4] << 32);

    Scalar fblock[4];
    decode_block_40(packed, fblock);

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
  compress_gather_10_kernel(
    const Scalar    *COMPRESSION_RESTRICT dataArray,
    const IndexType *COMPRESSION_RESTRICT indices,
    unsigned int                          gatherBlockSize,
    uint8_t *COMPRESSION_RESTRICT         stream,
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

    uint64   packed = encode_block_40(fblock);
    size_t   out    = (size_t)block_idx * 5u;
    stream[out]     = (uint8_t)(packed);
    stream[out + 1] = (uint8_t)(packed >> 8);
    stream[out + 2] = (uint8_t)(packed >> 16);
    stream[out + 3] = (uint8_t)(packed >> 24);
    stream[out + 4] = (uint8_t)(packed >> 32);
  }

  template <typename Scalar, typename IndexType>
  COMPRESSION_GLOBAL void
  decompress_scatter_add_10_kernel(
    const uint8_t   *COMPRESSION_RESTRICT stream,
    const IndexType *COMPRESSION_RESTRICT indices,
    unsigned int                          gatherBlockSize,
    Scalar *COMPRESSION_RESTRICT          dataArray,
    unsigned int                          tot_blocks)
  {
    const unsigned int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= tot_blocks)
      return;

    size_t out    = (size_t)block_idx * 5u;
    uint64 packed = (uint64)stream[out] |
                    ((uint64)stream[out + 1] << 8) |
                    ((uint64)stream[out + 2] << 16) |
                    ((uint64)stream[out + 3] << 24) |
                    ((uint64)stream[out + 4] << 32);

    Scalar fblock[4];
    decode_block_40(packed, fblock);

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
  compress_10_kernel(sycl::nd_item<1> item,
                         const Scalar    *data,
                         uint8_t         *stream,
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

    uint64   packed = encode_block_40(fblock);
    size_t   out    = (size_t)block_idx * 5u;
    stream[out]     = (uint8_t)(packed);
    stream[out + 1] = (uint8_t)(packed >> 8);
    stream[out + 2] = (uint8_t)(packed >> 16);
    stream[out + 3] = (uint8_t)(packed >> 24);
    stream[out + 4] = (uint8_t)(packed >> 32);
  }

  template <typename Scalar>
  void
  decompress_10_kernel(sycl::nd_item<1>  item,
                           const uint8_t    *stream,
                           Scalar           *data,
                           unsigned int      dim,
                           unsigned int      tot_blocks)
  {
    const unsigned int block_idx = item.get_global_id(0);
    if (block_idx >= tot_blocks)
      return;

    size_t out    = (size_t)block_idx * 5u;
    uint64 packed = (uint64)stream[out] |
                    ((uint64)stream[out + 1] << 8) |
                    ((uint64)stream[out + 2] << 16) |
                    ((uint64)stream[out + 3] << 24) |
                    ((uint64)stream[out + 4] << 32);

    Scalar fblock[4];
    decode_block_40(packed, fblock);

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
  compress_gather_10_kernel(sycl::nd_item<1>  item,
                                const Scalar     *dataArray,
                                const IndexType  *indices,
                                unsigned int      gatherBlockSize,
                                uint8_t          *stream,
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

    uint64   packed = encode_block_40(fblock);
    size_t   out    = (size_t)block_idx * 5u;
    stream[out]     = (uint8_t)(packed);
    stream[out + 1] = (uint8_t)(packed >> 8);
    stream[out + 2] = (uint8_t)(packed >> 16);
    stream[out + 3] = (uint8_t)(packed >> 24);
    stream[out + 4] = (uint8_t)(packed >> 32);
  }

  template <typename Scalar, typename IndexType>
  void
  decompress_scatter_add_10_kernel(sycl::nd_item<1>  item,
                                       const uint8_t    *stream,
                                       const IndexType  *indices,
                                       unsigned int      gatherBlockSize,
                                       Scalar           *dataArray,
                                       unsigned int      tot_blocks)
  {
    const unsigned int block_idx = item.get_global_id(0);
    if (block_idx >= tot_blocks)
      return;

    size_t out    = (size_t)block_idx * 5u;
    uint64 packed = (uint64)stream[out] |
                    ((uint64)stream[out + 1] << 8) |
                    ((uint64)stream[out + 2] << 16) |
                    ((uint64)stream[out + 3] << 24) |
                    ((uint64)stream[out + 4] << 32);

    Scalar fblock[4];
    decode_block_40(packed, fblock);

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

#endif /* BFP 10bpv specialized */

  /* =========================================================================
     Specialized BFP 8-bpv kernels (32-bit → 1 × uint32_t per block)

     Mirrors the 12-bpv design but with encode_block_32/decode_block_32
     (constexpr vbits/vmask). No runtime division or vmask computation
     in the hot path.

       block i → stream32[i]
     =========================================================================
   */

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)

  template <typename Scalar>
  COMPRESSION_GLOBAL void
  compress_8_kernel(const Scalar *COMPRESSION_RESTRICT data,
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

    stream[block_idx] = encode_block_32(fblock);
  }

  template <typename Scalar>
  COMPRESSION_GLOBAL void
  decompress_8_kernel(const unsigned int *COMPRESSION_RESTRICT stream,
                          Scalar *COMPRESSION_RESTRICT             data,
                          unsigned int                             dim,
                          unsigned int                             tot_blocks)
  {
    const unsigned int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= tot_blocks)
      return;

    Scalar fblock[4];
    decode_block_32(stream[block_idx], fblock);

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

  template <typename Scalar, typename IndexType>
  COMPRESSION_GLOBAL void
  compress_gather_8_kernel(
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

    stream[block_idx] = encode_block_32(fblock);
  }

  template <typename Scalar, typename IndexType>
  COMPRESSION_GLOBAL void
  decompress_scatter_add_8_kernel(
    const unsigned int *COMPRESSION_RESTRICT stream,
    const IndexType    *COMPRESSION_RESTRICT indices,
    unsigned int                             gatherBlockSize,
    Scalar *COMPRESSION_RESTRICT             dataArray,
    unsigned int                             tot_blocks)
  {
    const unsigned int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= tot_blocks)
      return;

    Scalar fblock[4];
    decode_block_32(stream[block_idx], fblock);

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

  template <typename Scalar>
  void
  compress_8_kernel(sycl::nd_item<1> item,
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

    stream[block_idx] = encode_block_32(fblock);
  }

  template <typename Scalar>
  void
  decompress_8_kernel(sycl::nd_item<1>    item,
                          const unsigned int *stream,
                          Scalar             *data,
                          unsigned int        dim,
                          unsigned int        tot_blocks)
  {
    const unsigned int block_idx = item.get_global_id(0);
    if (block_idx >= tot_blocks)
      return;

    Scalar fblock[4];
    decode_block_32(stream[block_idx], fblock);

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

  template <typename Scalar, typename IndexType>
  void
  compress_gather_8_kernel(sycl::nd_item<1>   item,
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

    stream[block_idx] = encode_block_32(fblock);
  }

  template <typename Scalar, typename IndexType>
  void
  decompress_scatter_add_8_kernel(
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

    Scalar fblock[4];
    decode_block_32(stream[block_idx], fblock);

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

#endif /* BFP 8bpv specialized */

  /* =========================================================================
     Specialized BFP 16-bpv kernels (64-bit → 1 × uint64 per block)

     1 thread per block, 1 × uint64 store. Uses encode_block_64/decode_block_64
     (constexpr vbits/vmask). All bit widths are compile-time constants.

       block i → stream64[i]
     =========================================================================
   */

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)

  template <typename Scalar>
  COMPRESSION_GLOBAL void
  compress_16_kernel(const Scalar *COMPRESSION_RESTRICT data,
                         uint64 *COMPRESSION_RESTRICT       stream,
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

    stream[block_idx] = encode_block_64(fblock);
  }

  template <typename Scalar>
  COMPRESSION_GLOBAL void
  decompress_16_kernel(const uint64 *COMPRESSION_RESTRICT stream,
                           Scalar *COMPRESSION_RESTRICT       data,
                           unsigned int                       dim,
                           unsigned int                       tot_blocks)
  {
    const unsigned int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= tot_blocks)
      return;

    Scalar fblock[4];
    decode_block_64(stream[block_idx], fblock);

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
  compress_gather_16_kernel(
    const Scalar    *COMPRESSION_RESTRICT dataArray,
    const IndexType *COMPRESSION_RESTRICT indices,
    unsigned int                          gatherBlockSize,
    uint64 *COMPRESSION_RESTRICT          stream,
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

    stream[block_idx] = encode_block_64(fblock);
  }

  template <typename Scalar, typename IndexType>
  COMPRESSION_GLOBAL void
  decompress_scatter_add_16_kernel(
    const uint64    *COMPRESSION_RESTRICT stream,
    const IndexType *COMPRESSION_RESTRICT indices,
    unsigned int                          gatherBlockSize,
    Scalar *COMPRESSION_RESTRICT          dataArray,
    unsigned int                          tot_blocks)
  {
    const unsigned int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= tot_blocks)
      return;

    Scalar fblock[4];
    decode_block_64(stream[block_idx], fblock);

    const unsigned int blocks_per_entry = gatherBlockSize >> 2;
    unsigned int       gatherIdx        = block_idx / blocks_per_entry;
    unsigned int       localBlock       = block_idx - gatherIdx * blocks_per_entry;
    unsigned int       intraIdx         = localBlock * 4u;
    size_t             base =
      (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

    portable_atomicAdd(&dataArray[base],     fblock[0]);
    portable_atomicAdd(&dataArray[base + 1], fblock[1]);
    portable_atomicAdd(&dataArray[base + 2], fblock[2]);
    portable_atomicAdd(&dataArray[base + 3], fblock[3]);
  }

#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)

  template <typename Scalar>
  void
  compress_16_kernel(sycl::nd_item<1> item,
                         const Scalar    *data,
                         uint64          *stream,
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

    stream[block_idx] = encode_block_64(fblock);
  }

  template <typename Scalar>
  void
  decompress_16_kernel(sycl::nd_item<1>  item,
                           const uint64     *stream,
                           Scalar           *data,
                           unsigned int      dim,
                           unsigned int      tot_blocks)
  {
    const unsigned int block_idx = item.get_global_id(0);
    if (block_idx >= tot_blocks)
      return;

    Scalar fblock[4];
    decode_block_64(stream[block_idx], fblock);

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
  compress_gather_16_kernel(sycl::nd_item<1>  item,
                                const Scalar     *dataArray,
                                const IndexType  *indices,
                                unsigned int      gatherBlockSize,
                                uint64           *stream,
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

    stream[block_idx] = encode_block_64(fblock);
  }

  template <typename Scalar, typename IndexType>
  void
  decompress_scatter_add_16_kernel(sycl::nd_item<1>  item,
                                       const uint64     *stream,
                                       const IndexType  *indices,
                                       unsigned int      gatherBlockSize,
                                       Scalar           *dataArray,
                                       unsigned int      tot_blocks)
  {
    const unsigned int block_idx = item.get_global_id(0);
    if (block_idx >= tot_blocks)
      return;

    Scalar fblock[4];
    decode_block_64(stream[block_idx], fblock);

    const unsigned int blocks_per_entry = gatherBlockSize >> 2;
    unsigned int       gatherIdx        = block_idx / blocks_per_entry;
    unsigned int       localBlock       = block_idx - gatherIdx * blocks_per_entry;
    unsigned int       intraIdx         = localBlock * 4u;
    size_t             base =
      (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

    portable_atomicAdd(&dataArray[base],     fblock[0]);
    portable_atomicAdd(&dataArray[base + 1], fblock[1]);
    portable_atomicAdd(&dataArray[base + 2], fblock[2]);
    portable_atomicAdd(&dataArray[base + 3], fblock[3]);
  }

#endif /* BFP 16bpv specialized */


} /* namespace compression */

#endif /* COMPRESSION_KERNELS_H */
