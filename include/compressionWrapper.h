#ifndef DFTFE_COMPRESSION_WRAPPER_H
#define DFTFE_COMPRESSION_WRAPPER_H

#ifdef DFTFE_WITH_DEVICE
#  include <cstddef>
#  include <complex>
#  include <DeviceTypeConfig.h>

namespace dftfe
{
  namespace compressionWrapper
  {
    void
    compress(const double                *d_data,
             void                        *d_compressed,
             size_t                       num_values,
             int                          bits_per_value,
             dftfe::utils::deviceStream_t stream);

    void
    compress(const float                 *d_data,
             void                        *d_compressed,
             size_t                       num_values,
             int                          bits_per_value,
             dftfe::utils::deviceStream_t stream);

    void
    decompress(const void                  *d_compressed,
               double                      *d_data,
               size_t                       num_values,
               int                          bits_per_value,
               dftfe::utils::deviceStream_t stream);

    void
    decompress(const void                  *d_compressed,
               float                       *d_data,
               size_t                       num_values,
               int                          bits_per_value,
               dftfe::utils::deviceStream_t stream);

    // BFP variants (faster, bpv <= 8)
    void
    compress_bfp(const double                *d_data,
                 void                        *d_compressed,
                 size_t                       num_values,
                 int                          bits_per_value,
                 dftfe::utils::deviceStream_t stream);

    void
    compress_bfp(const float                 *d_data,
                 void                        *d_compressed,
                 size_t                       num_values,
                 int                          bits_per_value,
                 dftfe::utils::deviceStream_t stream);

    void
    decompress_bfp(const void                  *d_compressed,
                   double                      *d_data,
                   size_t                       num_values,
                   int                          bits_per_value,
                   dftfe::utils::deviceStream_t stream);

    void
    decompress_bfp(const void                  *d_compressed,
                   float                       *d_data,
                   size_t                       num_values,
                   int                          bits_per_value,
                   dftfe::utils::deviceStream_t stream);

    // Complex overloads: treat complex<T>[N] as T[2*N]
    inline void
    compress(const std::complex<double>  *d_data,
             void                        *d_compressed,
             size_t                       num_values,
             int                          bits_per_value,
             dftfe::utils::deviceStream_t stream)
    {
      compress(reinterpret_cast<const double *>(d_data),
               d_compressed,
               num_values * 2,
               bits_per_value,
               stream);
    }

    inline void
    compress(const std::complex<float>   *d_data,
             void                        *d_compressed,
             size_t                       num_values,
             int                          bits_per_value,
             dftfe::utils::deviceStream_t stream)
    {
      compress(reinterpret_cast<const float *>(d_data),
               d_compressed,
               num_values * 2,
               bits_per_value,
               stream);
    }

    inline void
    decompress(const void                  *d_compressed,
               std::complex<double>        *d_data,
               size_t                       num_values,
               int                          bits_per_value,
               dftfe::utils::deviceStream_t stream)
    {
      decompress(d_compressed,
                 reinterpret_cast<double *>(d_data),
                 num_values * 2,
                 bits_per_value,
                 stream);
    }

    inline void
    decompress(const void                  *d_compressed,
               std::complex<float>         *d_data,
               size_t                       num_values,
               int                          bits_per_value,
               dftfe::utils::deviceStream_t stream)
    {
      decompress(d_compressed,
                 reinterpret_cast<float *>(d_data),
                 num_values * 2,
                 bits_per_value,
                 stream);
    }
    // BFP complex overloads
    inline void
    compress_bfp(const std::complex<double>  *d_data,
                 void                        *d_compressed,
                 size_t                       num_values,
                 int                          bits_per_value,
                 dftfe::utils::deviceStream_t stream)
    {
      compress_bfp(reinterpret_cast<const double *>(d_data),
                   d_compressed,
                   num_values * 2,
                   bits_per_value,
                   stream);
    }

    inline void
    compress_bfp(const std::complex<float>   *d_data,
                 void                        *d_compressed,
                 size_t                       num_values,
                 int                          bits_per_value,
                 dftfe::utils::deviceStream_t stream)
    {
      compress_bfp(reinterpret_cast<const float *>(d_data),
                   d_compressed,
                   num_values * 2,
                   bits_per_value,
                   stream);
    }

    inline void
    decompress_bfp(const void                  *d_compressed,
                   std::complex<double>        *d_data,
                   size_t                       num_values,
                   int                          bits_per_value,
                   dftfe::utils::deviceStream_t stream)
    {
      decompress_bfp(d_compressed,
                     reinterpret_cast<double *>(d_data),
                     num_values * 2,
                     bits_per_value,
                     stream);
    }

    inline void
    decompress_bfp(const void                  *d_compressed,
                   std::complex<float>         *d_data,
                   size_t                       num_values,
                   int                          bits_per_value,
                   dftfe::utils::deviceStream_t stream)
    {
      decompress_bfp(d_compressed,
                     reinterpret_cast<float *>(d_data),
                     num_values * 2,
                     bits_per_value,
                     stream);
    }
  } // namespace compressionWrapper
} // namespace dftfe

#endif // DFTFE_WITH_DEVICE
#endif // DFTFE_COMPRESSION_WRAPPER_H
