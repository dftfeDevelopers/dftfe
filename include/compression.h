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
**   - For 8, 10, 12, and 16 bpv: 1 thread per block (bpt=1), no atomicAdd,
**     no memset. Stores as uint32_t (8bpv), 5×uint8_t (10bpv),
**     3×uint16_t (12bpv), or uint64 (16bpv).
**     BFP uses constexpr vbits/vmask for all four rates.
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
** Internal structure (sub-headers, not for direct inclusion):
**   compressionTypes.h   – macros, typedefs, device helpers, traits,
**                          compressed_size, negabinary, lifting, quantise
**   compressionBlockIO.h – LocalBlockWriter/Reader, InlineBlockWriter/Reader,
**                          BlockReader
**   compressionZFP.h     – ZFP encode/decode core + all ZFP GPU kernels
**   compressionBFP.h     – BFP encode/decode helpers + all BFP GPU kernels
*/

#ifndef COMPRESSION_H
#define COMPRESSION_H

#include <compressionZFP.h>
#include <compressionBFP.h>

namespace compression
{

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
    else if (maxbits == 64u)
      {
        /* --- bpt=1 fast path (16 bpv): 1 thread/block, 1 × uint64 --- */
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        compress_zfp_64_kernel<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_data,
            reinterpret_cast<uint64 *>(d_stream),
            dim,
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compress_zfp_64_kernel<Scalar>),
                           grid,
                           COMPRESSION_BLOCK_SIZE,
                           0,
                           stream,
                           d_data,
                           reinterpret_cast<uint64 *>(d_stream),
                           dim,
                           num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto   &queue      = dftfe::utils::queueRegistry.find(stream)->second;
        uint64 *d_stream64 = reinterpret_cast<uint64 *>(d_stream);
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             compress_zfp_64_kernel<Scalar>(item,
                                                            d_data,
                                                            d_stream64,
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
    else if (maxbits == 64u)
      {
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
        const uint64 *d_stream64 =
          reinterpret_cast<const uint64 *>(d_stream);
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        decompress_zfp_64_kernel<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_stream64, d_data, dim, num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(decompress_zfp_64_kernel<Scalar>),
                           grid,
                           COMPRESSION_BLOCK_SIZE,
                           0,
                           stream,
                           d_stream64,
                           d_data,
                           dim,
                           num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto &queue = dftfe::utils::queueRegistry.find(stream)->second;
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             decompress_zfp_64_kernel<Scalar>(item,
                                                               d_stream64,
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
    else if (maxbits == 64u)
      {
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        compress_gather_zfp_64_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            dataArray,
            indices,
            gather_block_size,
            reinterpret_cast<uint64 *>(d_stream),
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(compress_gather_zfp_64_kernel<Scalar, IndexType>),
          grid,
          COMPRESSION_BLOCK_SIZE,
          0,
          stream,
          dataArray,
          indices,
          gather_block_size,
          reinterpret_cast<uint64 *>(d_stream),
          num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto   &queue      = dftfe::utils::queueRegistry.find(stream)->second;
        uint64 *d_stream64 = reinterpret_cast<uint64 *>(d_stream);
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             compress_gather_zfp_64_kernel<Scalar, IndexType>(
                               item,
                               dataArray,
                               indices,
                               gather_block_size,
                               d_stream64,
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
    else if (maxbits == 64u)
      {
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
        const uint64 *d_stream64 =
          reinterpret_cast<const uint64 *>(d_stream);
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        decompress_scatter_add_zfp_64_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_stream64,
            indices,
            gather_block_size,
            dataArray,
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
            decompress_scatter_add_zfp_64_kernel<Scalar, IndexType>),
          grid,
          COMPRESSION_BLOCK_SIZE,
          0,
          stream,
          d_stream64,
          indices,
          gather_block_size,
          dataArray,
          num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto &queue = dftfe::utils::queueRegistry.find(stream)->second;
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             decompress_scatter_add_zfp_64_kernel<Scalar,
                                                                  IndexType>(
                               item,
                               d_stream64,
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
   *   bpv == 8:   uint32_t fast path (1 thread/block, 1 store)
   *   bpv == 10:  5×uint8_t fast path (1 thread/block, 5 byte stores)
   *   bpv == 12:  3×uint16_t fast path (1 thread/block, 3 half-word stores)
   *   bpv == 16:  uint64 fast path (1 thread/block, 1 store)
   *   other bpv:  super-block path (same layout as ZFP, LocalBlockWriter)
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
        /* --- specialized 32-bit path (bpv == 8): constexpr vbits/vmask --- */
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        compress_bfp_8_kernel<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_data,
            reinterpret_cast<unsigned int *>(d_stream),
            dim,
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compress_bfp_8_kernel<Scalar>),
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
                             compress_bfp_8_kernel<Scalar>(item,
                                                           d_data,
                                                           d_stream_u32,
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
    else if (maxbits == 40u)
      {
        /* --- specialized 40-bit path (bpv == 10): 1 thread/block, 5×uint8 --- */
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        compress_bfp_10_kernel<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_data,
            reinterpret_cast<uint8_t *>(d_stream),
            dim,
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compress_bfp_10_kernel<Scalar>),
                           grid,
                           COMPRESSION_BLOCK_SIZE,
                           0,
                           stream,
                           d_data,
                           reinterpret_cast<uint8_t *>(d_stream),
                           dim,
                           num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto    &queue     = dftfe::utils::queueRegistry.find(stream)->second;
        uint8_t *d_stream8 = reinterpret_cast<uint8_t *>(d_stream);
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             compress_bfp_10_kernel<Scalar>(item,
                                                            d_data,
                                                            d_stream8,
                                                            dim,
                                                            num_blocks);
                           });
#endif
      }
    else if (maxbits == 64u)
      {
        /* --- specialized 64-bit path (bpv == 16): 1 thread/block, 1×uint64 --- */
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        compress_bfp_16_kernel<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_data,
            reinterpret_cast<uint64 *>(d_stream),
            dim,
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compress_bfp_16_kernel<Scalar>),
                           grid,
                           COMPRESSION_BLOCK_SIZE,
                           0,
                           stream,
                           d_data,
                           reinterpret_cast<uint64 *>(d_stream),
                           dim,
                           num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto   &queue      = dftfe::utils::queueRegistry.find(stream)->second;
        uint64 *d_stream64 = reinterpret_cast<uint64 *>(d_stream);
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             compress_bfp_16_kernel<Scalar>(item,
                                                            d_data,
                                                            d_stream64,
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
   *   bpv == 8:   uint32_t fast path
   *   bpv == 10:  5×uint8_t fast path
   *   bpv == 12:  3×uint16_t fast path
   *   bpv == 16:  uint64 fast path
   *   other bpv:  BlockReader path (reads from packed Word stream)
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
        /* --- specialized 32-bit path (bpv == 8): constexpr vbits/vmask --- */
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
        const unsigned int *d_stream_u32 =
          reinterpret_cast<const unsigned int *>(d_stream);
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        decompress_bfp_8_kernel<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_stream_u32, d_data, dim, num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(decompress_bfp_8_kernel<Scalar>),
                           grid,
                           COMPRESSION_BLOCK_SIZE,
                           0,
                           stream,
                           d_stream_u32,
                           d_data,
                           dim,
                           num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto &queue = dftfe::utils::queueRegistry.find(stream)->second;
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             decompress_bfp_8_kernel<Scalar>(item,
                                                             d_stream_u32,
                                                             d_data,
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
    else if (maxbits == 40u)
      {
        /* --- specialized 40-bit path (bpv == 10): 1 thread/block, 5×uint8 --- */
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
        const uint8_t *d_stream8 =
          reinterpret_cast<const uint8_t *>(d_stream);
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        decompress_bfp_10_kernel<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_stream8, d_data, dim, num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(decompress_bfp_10_kernel<Scalar>),
                           grid,
                           COMPRESSION_BLOCK_SIZE,
                           0,
                           stream,
                           d_stream8,
                           d_data,
                           dim,
                           num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto &queue = dftfe::utils::queueRegistry.find(stream)->second;
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             decompress_bfp_10_kernel<Scalar>(item,
                                                              d_stream8,
                                                              d_data,
                                                              dim,
                                                              num_blocks);
                           });
#endif
      }
    else if (maxbits == 64u)
      {
        /* --- specialized 64-bit path (bpv == 16): 1 thread/block, 1×uint64 --- */
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
        const uint64 *d_stream64 =
          reinterpret_cast<const uint64 *>(d_stream);
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        decompress_bfp_16_kernel<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_stream64, d_data, dim, num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(decompress_bfp_16_kernel<Scalar>),
                           grid,
                           COMPRESSION_BLOCK_SIZE,
                           0,
                           stream,
                           d_stream64,
                           d_data,
                           dim,
                           num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto &queue = dftfe::utils::queueRegistry.find(stream)->second;
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             decompress_bfp_16_kernel<Scalar>(item,
                                                              d_stream64,
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
   * directly. Fast paths for bpv == 8/10/12/16; super-block path otherwise.
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
        compress_gather_bfp_8_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            dataArray,
            indices,
            gather_block_size,
            reinterpret_cast<unsigned int *>(d_stream),
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(compress_gather_bfp_8_kernel<Scalar, IndexType>),
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
                             compress_gather_bfp_8_kernel<Scalar, IndexType>(
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
    else if (maxbits == 40u)
      {
        /* --- specialized 40-bit path (bpv == 10): 1 thread/block, 5×uint8 --- */
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        compress_gather_bfp_10_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            dataArray,
            indices,
            gather_block_size,
            reinterpret_cast<uint8_t *>(d_stream),
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(compress_gather_bfp_10_kernel<Scalar, IndexType>),
          grid,
          COMPRESSION_BLOCK_SIZE,
          0,
          stream,
          dataArray,
          indices,
          gather_block_size,
          reinterpret_cast<uint8_t *>(d_stream),
          num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto    &queue     = dftfe::utils::queueRegistry.find(stream)->second;
        uint8_t *d_stream8 = reinterpret_cast<uint8_t *>(d_stream);
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             compress_gather_bfp_10_kernel<Scalar, IndexType>(
                               item,
                               dataArray,
                               indices,
                               gather_block_size,
                               d_stream8,
                               num_blocks);
                           });
#endif
      }
    else if (maxbits == 64u)
      {
        /* --- specialized 64-bit path (bpv == 16): 1 thread/block, 1×uint64 --- */
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        compress_gather_bfp_16_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            dataArray,
            indices,
            gather_block_size,
            reinterpret_cast<uint64 *>(d_stream),
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(compress_gather_bfp_16_kernel<Scalar, IndexType>),
          grid,
          COMPRESSION_BLOCK_SIZE,
          0,
          stream,
          dataArray,
          indices,
          gather_block_size,
          reinterpret_cast<uint64 *>(d_stream),
          num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto   &queue      = dftfe::utils::queueRegistry.find(stream)->second;
        uint64 *d_stream64 = reinterpret_cast<uint64 *>(d_stream);
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             compress_gather_bfp_16_kernel<Scalar, IndexType>(
                               item,
                               dataArray,
                               indices,
                               gather_block_size,
                               d_stream64,
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
   * to scattered positions in dataArray. Fast paths for bpv == 8/10/12/16.
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
        decompress_scatter_add_bfp_8_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_stream_u32,
            indices,
            gather_block_size,
            dataArray,
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(decompress_scatter_add_bfp_8_kernel<Scalar, IndexType>),
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
                             decompress_scatter_add_bfp_8_kernel<Scalar,
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
    else if (maxbits == 40u)
      {
        /* --- specialized 40-bit path (bpv == 10): 1 thread/block, 5×uint8 --- */
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
        const uint8_t *d_stream8 =
          reinterpret_cast<const uint8_t *>(d_stream);
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        decompress_scatter_add_bfp_10_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_stream8,
            indices,
            gather_block_size,
            dataArray,
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
            decompress_scatter_add_bfp_10_kernel<Scalar, IndexType>),
          grid,
          COMPRESSION_BLOCK_SIZE,
          0,
          stream,
          d_stream8,
          indices,
          gather_block_size,
          dataArray,
          num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto &queue = dftfe::utils::queueRegistry.find(stream)->second;
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             decompress_scatter_add_bfp_10_kernel<Scalar,
                                                                  IndexType>(
                               item,
                               d_stream8,
                               indices,
                               gather_block_size,
                               dataArray,
                               num_blocks);
                           });
#endif
      }
    else if (maxbits == 64u)
      {
        /* --- specialized 64-bit path (bpv == 16): 1 thread/block, 1×uint64 --- */
        const unsigned int grid =
          (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;
        const uint64 *d_stream64 =
          reinterpret_cast<const uint64 *>(d_stream);
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        decompress_scatter_add_bfp_16_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_stream64,
            indices,
            gather_block_size,
            dataArray,
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
            decompress_scatter_add_bfp_16_kernel<Scalar, IndexType>),
          grid,
          COMPRESSION_BLOCK_SIZE,
          0,
          stream,
          d_stream64,
          indices,
          gather_block_size,
          dataArray,
          num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        auto &queue = dftfe::utils::queueRegistry.find(stream)->second;
        queue.parallel_for(sycl::nd_range<1>(grid * COMPRESSION_BLOCK_SIZE,
                                             COMPRESSION_BLOCK_SIZE),
                           [=](sycl::nd_item<1> item) {
                             decompress_scatter_add_bfp_16_kernel<Scalar,
                                                                  IndexType>(
                               item,
                               d_stream64,
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
