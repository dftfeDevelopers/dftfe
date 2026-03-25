/*
** compressionZFP.h - GPU compressor: ZFP-style fixed-rate encode/decode.
**
**   Core algorithm (templated on Writer/Reader):
**     encode_block_ints, encode_block, decode_ints, decode_block
**
**   GPU kernels:
**     Super-block (generic bpv):
**       compress_kernel, decompress_kernel_sb
**       compress_gather_kernel, decompress_scatter_add_kernel
**
**     bpt=1 specializations:
**       8  bpv (uint32_t stream):  compress_zfp_32_kernel, decompress_zfp_32_kernel,
**                                  compress_gather_zfp_32_kernel, decompress_scatter_add_zfp_32_kernel
**       12 bpv (3×uint16_t):       compress_zfp_48_kernel, decompress_zfp_48_kernel,
**                                  compress_gather_zfp_48_kernel, decompress_scatter_add_zfp_48_kernel
**       16 bpv (uint64 stream):    compress_zfp_64_kernel, decompress_zfp_64_kernel,
**                                  compress_gather_zfp_64_kernel, decompress_scatter_add_zfp_64_kernel
**
** CUDA/HIP and SYCL backends are both included via preprocessor guards.
** Included by compression.h. Never included directly by application code.
*/

#ifndef COMPRESSION_ZFP_H
#define COMPRESSION_ZFP_H

#include <compressionBlockIO.h>

namespace compression
{

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

  /* =========================================================================
     Specialized ZFP 16-bpv (64-bit) kernels

     1 thread per block, 1 × uint64 store. Mirrors the 8-bpv uint32_t and
     12-bpv 3×uint16_t designs: each thread independently encodes one block
     using InlineBlockWriter (direct uint64 accumulation), eliminating the
     LocalBlockWriter Word-array, modulo/division, and cross-word branches.

       16 bpv (maxbits=64): each block → 1 × uint64   (bpt=1, wpt=1)
     =========================================================================
   */

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)

  /* --- ZFP 16bpv: 1 thread/block, 1 × uint64 --- */

  template <typename Scalar>
  COMPRESSION_GLOBAL void
  compress_zfp_64_kernel(const Scalar *COMPRESSION_RESTRICT data,
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

    InlineBlockWriter writer;
    encode_block(writer, fblock, 64u);
    stream[block_idx] = writer.m_packed;
  }

  template <typename Scalar>
  COMPRESSION_GLOBAL void
  decompress_zfp_64_kernel(const uint64 *COMPRESSION_RESTRICT stream,
                           Scalar *COMPRESSION_RESTRICT       data,
                           unsigned int                       dim,
                           unsigned int                       tot_blocks)
  {
    const unsigned int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= tot_blocks)
      return;

    InlineBlockReader reader(stream[block_idx]);
    Scalar            fblock[4];
    decode_block<InlineBlockReader, Scalar>(reader, fblock, 64u);

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
  compress_gather_zfp_64_kernel(
    const Scalar    *COMPRESSION_RESTRICT dataArray,
    const IndexType *COMPRESSION_RESTRICT indices,
    unsigned int                          gatherBlockSize,
    uint64 *COMPRESSION_RESTRICT          stream,
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
    encode_block(writer, fblock, 64u);
    stream[block_idx] = writer.m_packed;
  }

  template <typename Scalar, typename IndexType>
  COMPRESSION_GLOBAL void
  decompress_scatter_add_zfp_64_kernel(
    const uint64    *COMPRESSION_RESTRICT stream,
    const IndexType *COMPRESSION_RESTRICT indices,
    unsigned int                          gatherBlockSize,
    Scalar *COMPRESSION_RESTRICT          dataArray,
    unsigned int                          tot_blocks)
  {
    const unsigned int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= tot_blocks)
      return;

    InlineBlockReader reader(stream[block_idx]);
    Scalar            fblock[4];
    decode_block<InlineBlockReader, Scalar>(reader, fblock, 64u);

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

  /* --- ZFP 16bpv SYCL: 1 thread/block, 1 × uint64 --- */

  template <typename Scalar>
  void
  compress_zfp_64_kernel(sycl::nd_item<1> item,
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

    InlineBlockWriter writer;
    encode_block(writer, fblock, 64u);
    stream[block_idx] = writer.m_packed;
  }

  template <typename Scalar>
  void
  decompress_zfp_64_kernel(sycl::nd_item<1>  item,
                           const uint64     *stream,
                           Scalar           *data,
                           unsigned int      dim,
                           unsigned int      tot_blocks)
  {
    const unsigned int block_idx = item.get_global_id(0);
    if (block_idx >= tot_blocks)
      return;

    InlineBlockReader reader(stream[block_idx]);
    Scalar            fblock[4];
    decode_block<InlineBlockReader, Scalar>(reader, fblock, 64u);

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
  compress_gather_zfp_64_kernel(sycl::nd_item<1>  item,
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
    encode_block(writer, fblock, 64u);
    stream[block_idx] = writer.m_packed;
  }

  template <typename Scalar, typename IndexType>
  void
  decompress_scatter_add_zfp_64_kernel(
    sycl::nd_item<1>    item,
    const uint64       *stream,
    const IndexType    *indices,
    unsigned int        gatherBlockSize,
    Scalar             *dataArray,
    unsigned int        tot_blocks)
  {
    const unsigned int block_idx = item.get_global_id(0);
    if (block_idx >= tot_blocks)
      return;

    InlineBlockReader reader(stream[block_idx]);
    Scalar            fblock[4];
    decode_block<InlineBlockReader, Scalar>(reader, fblock, 64u);

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

#endif /* ZFP 16bpv specialized kernels */

#endif /* ZFP specialized bpt=1 kernels */

} /* namespace compression */

#endif /* COMPRESSION_ZFP_H */
