/*
** compression.h - GPU Fixed-Rate Floating-Point Compressor
**
** Specialized for:
**   - 1D contiguous arrays only (stride = 1)
**   - Fixed-rate Block Floating Point (BFP), bits_per_value in {8, 10, 12, 16}
**   - GPU-resident data (zero host<->device copies)
**   - User-specified device stream (non-blocking, async)
**   - float and double types
**   - Portable across CUDA, HIP, and SYCL backends
**
** Performance:
**   1 thread per block (bpt=1), no atomicAdd, no memset.  Each block of
**   four values packs into a fixed-width word: shared exponent + 4 uniform
**   fixed-point quantized coefficients. Stores per block:
**     bpv == 8:  uint32_t        (32 bits)
**     bpv == 10: 5 x uint8_t     (40 bits)
**     bpv == 12: 3 x uint16_t    (48 bits)
**     bpv == 16: uint64          (64 bits)
**   constexpr vbits/vmask for all four rates.
**
** Usage:
**   #include "compression.h"
**
**   compression::compress(d_in, d_comp, N, 12, my_stream);
**   // ... NCCL/MPI send d_comp ...
**   compression::decompress(d_comp, d_out, N, 12, my_stream);
**
** References:
* [1] P. Lindstrom, "Fixed-Rate Compressed Floating-Point Arrays," IEEE Trans.
*     Vis. Comput. Graph., vol. 20, no. 12, pp. 2674-2683, Dec. 2014.
*     DOI: 10.1109/TVCG.2014.2346458
**
** Internal structure (sub-headers, not for direct inclusion):
**   compressionTypes.h    – macros, typedefs, device helpers, traits, pad_block
**   compressionKernels.h  – BFP encode/decode helpers + GPU kernels
*/

#ifndef COMPRESSION_H
#define COMPRESSION_H

#include <compressionKernels.h>

namespace compression
{

#define COMPRESSION_ASSERT_BPV(bpv)                              \
  assert(((bpv) == 8 || (bpv) == 10 || (bpv) == 12 || (bpv) == 16) && \
         "bits_per_value must be 8, 10, 12, or 16")

  /*
   * Compress — fixed-rate BFP.  bits_per_value in {8, 10, 12, 16}.
   * Buffer size: ceil(N * bits_per_value / 8) bytes (rounded up to 8).
   */
  template <typename Scalar>
  inline void
  compress(
    const Scalar                *d_data,
    void                        *d_stream,
    size_t                       num_values,
    int                          bits_per_value,
    dftfe::utils::deviceStream_t stream = dftfe::utils::defaultStream)
  {
    if (num_values == 0)
      return;
    assert(num_values <= (size_t)UINT_MAX && "num_values exceeds 32-bit limit");
    COMPRESSION_ASSERT_BPV(bits_per_value);

    const unsigned int dim        = (unsigned int)num_values;
    const unsigned int num_blocks = (dim + 3u) / 4u;
    const unsigned int maxbits    = (unsigned int)bits_per_value * 4u;
    const unsigned int grid =
      (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;

    if (maxbits == 32u)
      {
        /* bpv == 8: uint32_t per block */
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        compress_8_kernel<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_data,
            reinterpret_cast<unsigned int *>(d_stream),
            dim,
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compress_8_kernel<Scalar>),
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
                             compress_8_kernel<Scalar>(item,
                                                       d_data,
                                                       d_stream_u32,
                                                       dim,
                                                       num_blocks);
                           });
#endif
      }
    else if (maxbits == 40u)
      {
        /* bpv == 10: 5 x uint8_t per block */
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        compress_10_kernel<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_data,
            reinterpret_cast<uint8_t *>(d_stream),
            dim,
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compress_10_kernel<Scalar>),
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
                             compress_10_kernel<Scalar>(item,
                                                        d_data,
                                                        d_stream8,
                                                        dim,
                                                        num_blocks);
                           });
#endif
      }
    else if (maxbits == 48u)
      {
        /* bpv == 12: 3 x uint16_t per block */
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        compress_12_kernel<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_data,
            reinterpret_cast<uint16_t *>(d_stream),
            dim,
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compress_12_kernel<Scalar>),
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
                             compress_12_kernel<Scalar>(item,
                                                        d_data,
                                                        d_stream16,
                                                        dim,
                                                        num_blocks);
                           });
#endif
      }
    else /* maxbits == 64u, bpv == 16 */
      {
        /* bpv == 16: uint64 per block */
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        compress_16_kernel<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_data,
            reinterpret_cast<uint64 *>(d_stream),
            dim,
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compress_16_kernel<Scalar>),
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
                             compress_16_kernel<Scalar>(item,
                                                        d_data,
                                                        d_stream64,
                                                        dim,
                                                        num_blocks);
                           });
#endif
      }
  }

  /*
   * Decompress — fixed-rate BFP.  bits_per_value in {8, 10, 12, 16}.
   */
  template <typename Scalar>
  inline void
  decompress(
    const void                  *d_stream,
    Scalar                      *d_data,
    size_t                       num_values,
    int                          bits_per_value,
    dftfe::utils::deviceStream_t stream = dftfe::utils::defaultStream)
  {
    if (num_values == 0)
      return;
    assert(num_values <= (size_t)UINT_MAX && "num_values exceeds 32-bit limit");
    COMPRESSION_ASSERT_BPV(bits_per_value);

    const unsigned int dim        = (unsigned int)num_values;
    const unsigned int num_blocks = (dim + 3u) / 4u;
    const unsigned int maxbits    = (unsigned int)bits_per_value * 4u;
    const unsigned int grid =
      (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;

    if (maxbits == 32u)
      {
        const unsigned int *d_stream_u32 =
          reinterpret_cast<const unsigned int *>(d_stream);
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        decompress_8_kernel<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_stream_u32, d_data, dim, num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(decompress_8_kernel<Scalar>),
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
                             decompress_8_kernel<Scalar>(item,
                                                         d_stream_u32,
                                                         d_data,
                                                         dim,
                                                         num_blocks);
                           });
#endif
      }
    else if (maxbits == 40u)
      {
        const uint8_t *d_stream8 =
          reinterpret_cast<const uint8_t *>(d_stream);
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        decompress_10_kernel<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_stream8, d_data, dim, num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(decompress_10_kernel<Scalar>),
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
                             decompress_10_kernel<Scalar>(item,
                                                          d_stream8,
                                                          d_data,
                                                          dim,
                                                          num_blocks);
                           });
#endif
      }
    else if (maxbits == 48u)
      {
        const uint16_t *d_stream16 =
          reinterpret_cast<const uint16_t *>(d_stream);
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        decompress_12_kernel<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_stream16, d_data, dim, num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(decompress_12_kernel<Scalar>),
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
                             decompress_12_kernel<Scalar>(item,
                                                          d_stream16,
                                                          d_data,
                                                          dim,
                                                          num_blocks);
                           });
#endif
      }
    else /* maxbits == 64u, bpv == 16 */
      {
        const uint64 *d_stream64 =
          reinterpret_cast<const uint64 *>(d_stream);
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        decompress_16_kernel<Scalar>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_stream64, d_data, dim, num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(decompress_16_kernel<Scalar>),
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
                             decompress_16_kernel<Scalar>(item,
                                                          d_stream64,
                                                          d_data,
                                                          dim,
                                                          num_blocks);
                           });
#endif
      }
  }

  /*
   * Fused gather+compress: reads scattered data via indices and compresses
   * directly. bits_per_value in {8, 10, 12, 16}.
   * Assumes gather_block_size is a multiple of 4.
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
    COMPRESSION_ASSERT_BPV(bits_per_value);

    const unsigned int dim        = (unsigned int)num_values;
    const unsigned int num_blocks = (dim + 3u) / 4u;
    const unsigned int maxbits    = (unsigned int)bits_per_value * 4u;
    const unsigned int grid =
      (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;

    if (maxbits == 32u)
      {
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        compress_gather_8_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            dataArray,
            indices,
            gather_block_size,
            reinterpret_cast<unsigned int *>(d_stream),
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(compress_gather_8_kernel<Scalar, IndexType>),
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
                             compress_gather_8_kernel<Scalar, IndexType>(
                               item,
                               dataArray,
                               indices,
                               gather_block_size,
                               d_stream_u32,
                               num_blocks);
                           });
#endif
      }
    else if (maxbits == 40u)
      {
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        compress_gather_10_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            dataArray,
            indices,
            gather_block_size,
            reinterpret_cast<uint8_t *>(d_stream),
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(compress_gather_10_kernel<Scalar, IndexType>),
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
                             compress_gather_10_kernel<Scalar, IndexType>(
                               item,
                               dataArray,
                               indices,
                               gather_block_size,
                               d_stream8,
                               num_blocks);
                           });
#endif
      }
    else if (maxbits == 48u)
      {
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        compress_gather_12_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            dataArray,
            indices,
            gather_block_size,
            reinterpret_cast<uint16_t *>(d_stream),
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(compress_gather_12_kernel<Scalar, IndexType>),
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
                             compress_gather_12_kernel<Scalar, IndexType>(
                               item,
                               dataArray,
                               indices,
                               gather_block_size,
                               d_stream16,
                               num_blocks);
                           });
#endif
      }
    else /* maxbits == 64u, bpv == 16 */
      {
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        compress_gather_16_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            dataArray,
            indices,
            gather_block_size,
            reinterpret_cast<uint64 *>(d_stream),
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(compress_gather_16_kernel<Scalar, IndexType>),
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
                             compress_gather_16_kernel<Scalar, IndexType>(
                               item,
                               dataArray,
                               indices,
                               gather_block_size,
                               d_stream64,
                               num_blocks);
                           });
#endif
      }
  }

  /*
   * Fused decompress+scatter_add: decompresses and atomicAdds directly to
   * scattered positions in dataArray. bits_per_value in {8, 10, 12, 16}.
   * Assumes gather_block_size is a multiple of 4.
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
    COMPRESSION_ASSERT_BPV(bits_per_value);

    const unsigned int dim        = (unsigned int)num_values;
    const unsigned int num_blocks = (dim + 3u) / 4u;
    const unsigned int maxbits    = (unsigned int)bits_per_value * 4u;
    const unsigned int grid =
      (num_blocks + COMPRESSION_BLOCK_SIZE - 1) / COMPRESSION_BLOCK_SIZE;

    if (maxbits == 32u)
      {
        const unsigned int *d_stream_u32 =
          reinterpret_cast<const unsigned int *>(d_stream);
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        decompress_scatter_add_8_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_stream_u32,
            indices,
            gather_block_size,
            dataArray,
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(decompress_scatter_add_8_kernel<Scalar, IndexType>),
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
                             decompress_scatter_add_8_kernel<Scalar,
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
    else if (maxbits == 40u)
      {
        const uint8_t *d_stream8 =
          reinterpret_cast<const uint8_t *>(d_stream);
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        decompress_scatter_add_10_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_stream8,
            indices,
            gather_block_size,
            dataArray,
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
            decompress_scatter_add_10_kernel<Scalar, IndexType>),
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
                             decompress_scatter_add_10_kernel<Scalar,
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
    else if (maxbits == 48u)
      {
        const uint16_t *d_stream16 =
          reinterpret_cast<const uint16_t *>(d_stream);
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        decompress_scatter_add_12_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_stream16,
            indices,
            gather_block_size,
            dataArray,
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
            decompress_scatter_add_12_kernel<Scalar, IndexType>),
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
                             decompress_scatter_add_12_kernel<Scalar,
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
    else /* maxbits == 64u, bpv == 16 */
      {
        const uint64 *d_stream64 =
          reinterpret_cast<const uint64 *>(d_stream);
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
        decompress_scatter_add_16_kernel<Scalar, IndexType>
          <<<grid, COMPRESSION_BLOCK_SIZE, 0, stream>>>(
            d_stream64,
            indices,
            gather_block_size,
            dataArray,
            num_blocks);
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
            decompress_scatter_add_16_kernel<Scalar, IndexType>),
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
                             decompress_scatter_add_16_kernel<Scalar,
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
  }

#undef COMPRESSION_ASSERT_BPV

} /* namespace compression */

#endif /* COMPRESSION_H */
