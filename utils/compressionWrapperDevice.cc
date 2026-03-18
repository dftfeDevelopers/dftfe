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
    // BFP variants
    void
    compress_bfp(const double                *d_data,
                 void                        *d_compressed,
                 size_t                       num_values,
                 int                          bits_per_value,
                 dftfe::utils::deviceStream_t stream)
    {
      ::compression::compress_bfp<double>(
        d_data, d_compressed, num_values, bits_per_value, stream);
    }

    void
    compress_bfp(const float                 *d_data,
                 void                        *d_compressed,
                 size_t                       num_values,
                 int                          bits_per_value,
                 dftfe::utils::deviceStream_t stream)
    {
      ::compression::compress_bfp<float>(
        d_data, d_compressed, num_values, bits_per_value, stream);
    }

    void
    decompress_bfp(const void                  *d_compressed,
                   double                      *d_data,
                   size_t                       num_values,
                   int                          bits_per_value,
                   dftfe::utils::deviceStream_t stream)
    {
      ::compression::decompress_bfp<double>(
        d_compressed, d_data, num_values, bits_per_value, stream);
    }

    void
    decompress_bfp(const void                  *d_compressed,
                   float                       *d_data,
                   size_t                       num_values,
                   int                          bits_per_value,
                   dftfe::utils::deviceStream_t stream)
    {
      ::compression::decompress_bfp<float>(
        d_compressed, d_data, num_values, bits_per_value, stream);
    }
  } // namespace compressionWrapper
} // namespace dftfe

#endif // DFTFE_WITH_DEVICE
