#include <compression.h>
#include <compressionWrapper.h>

#ifdef DFTFE_WITH_DEVICE

namespace dftfe
{
  namespace compressionWrapper
  {
    void
    compress(const double                *d_data,
             void                        *d_compressed,
             size_t                       num_values,
             int                          bits_per_value,
             dftfe::utils::deviceStream_t stream)
    {
      ::compression::compress<double>(
        d_data, d_compressed, num_values, bits_per_value, stream);
    }

    void
    compress(const float                 *d_data,
             void                        *d_compressed,
             size_t                       num_values,
             int                          bits_per_value,
             dftfe::utils::deviceStream_t stream)
    {
      ::compression::compress<float>(
        d_data, d_compressed, num_values, bits_per_value, stream);
    }

    void
    decompress(const void                  *d_compressed,
               double                      *d_data,
               size_t                       num_values,
               int                          bits_per_value,
               dftfe::utils::deviceStream_t stream)
    {
      ::compression::decompress<double>(
        d_compressed, d_data, num_values, bits_per_value, stream);
    }

    void
    decompress(const void                  *d_compressed,
               float                       *d_data,
               size_t                       num_values,
               int                          bits_per_value,
               dftfe::utils::deviceStream_t stream)
    {
      ::compression::decompress<float>(
        d_compressed, d_data, num_values, bits_per_value, stream);
    }

    // Fused gather+compress
    void
    compress_gather(const double                *dataArray,
                    const dftfe::uInt           *indices,
                    size_t                       num_indices,
                    dftfe::uInt                  gather_block_size,
                    void                        *d_compressed,
                    int                          bits_per_value,
                    dftfe::utils::deviceStream_t stream)
    {
      ::compression::compress_gather<double>(
        dataArray, indices, num_indices, gather_block_size,
        d_compressed, bits_per_value, stream);
    }

    void
    compress_gather(const float                 *dataArray,
                    const dftfe::uInt           *indices,
                    size_t                       num_indices,
                    dftfe::uInt                  gather_block_size,
                    void                        *d_compressed,
                    int                          bits_per_value,
                    dftfe::utils::deviceStream_t stream)
    {
      ::compression::compress_gather<float>(
        dataArray, indices, num_indices, gather_block_size,
        d_compressed, bits_per_value, stream);
    }

    // Fused decompress+scatter_add
    void
    decompress_scatter_add(const void                  *d_compressed,
                           const dftfe::uInt           *indices,
                           size_t                       num_indices,
                           dftfe::uInt                  gather_block_size,
                           double                      *dataArray,
                           int                          bits_per_value,
                           dftfe::utils::deviceStream_t stream)
    {
      ::compression::decompress_scatter_add<double>(
        d_compressed, indices, num_indices, gather_block_size,
        dataArray, bits_per_value, stream);
    }

    void
    decompress_scatter_add(const void                  *d_compressed,
                           const dftfe::uInt           *indices,
                           size_t                       num_indices,
                           dftfe::uInt                  gather_block_size,
                           float                       *dataArray,
                           int                          bits_per_value,
                           dftfe::utils::deviceStream_t stream)
    {
      ::compression::decompress_scatter_add<float>(
        d_compressed, indices, num_indices, gather_block_size,
        dataArray, bits_per_value, stream);
    }

  } // namespace compressionWrapper
} // namespace dftfe

#endif // DFTFE_WITH_DEVICE
