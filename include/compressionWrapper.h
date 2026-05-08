#ifndef DFTFE_COMPRESSION_WRAPPER_H
#define DFTFE_COMPRESSION_WRAPPER_H

#ifdef DFTFE_WITH_DEVICE
#  include <cstddef>
#  include <complex>
#  include <TypeConfig.h>
#  include <DeviceTypeConfig.h>

namespace dftfe
{
  namespace compressionWrapper
  {
    // BFP compression is only used at runtime for float / std::complex<float>
    // (the COMPRESSED communication-precision branch is set on FP32
    // multivectors only). The double / complex<double> overloads exist solely
    // because MPICommunicatorP2P is explicitly instantiated for those types
    // and its compress branch is a runtime if (d_commPrecision == compress) —
    // the compiler still has to resolve those call sites at instantiation
    // time. The double-typed overloads link but are never executed.

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

    // Fused gather+compress
    void
    compress_gather(const double                *dataArray,
                    const dftfe::uInt           *indices,
                    size_t                       num_indices,
                    dftfe::uInt                  gather_block_size,
                    void                        *d_compressed,
                    int                          bits_per_value,
                    dftfe::utils::deviceStream_t stream);

    void
    compress_gather(const float                 *dataArray,
                    const dftfe::uInt           *indices,
                    size_t                       num_indices,
                    dftfe::uInt                  gather_block_size,
                    void                        *d_compressed,
                    int                          bits_per_value,
                    dftfe::utils::deviceStream_t stream);

    // Fused decompress+scatter_add
    void
    decompress_scatter_add(const void                  *d_compressed,
                           const dftfe::uInt           *indices,
                           size_t                       num_indices,
                           dftfe::uInt                  gather_block_size,
                           double                      *dataArray,
                           int                          bits_per_value,
                           dftfe::utils::deviceStream_t stream);

    void
    decompress_scatter_add(const void                  *d_compressed,
                           const dftfe::uInt           *indices,
                           size_t                       num_indices,
                           dftfe::uInt                  gather_block_size,
                           float                       *dataArray,
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

    // Fused gather+compress complex overloads
    inline void
    compress_gather(const std::complex<double>  *dataArray,
                    const dftfe::uInt           *indices,
                    size_t                       num_indices,
                    dftfe::uInt                  gather_block_size,
                    void                        *d_compressed,
                    int                          bits_per_value,
                    dftfe::utils::deviceStream_t stream)
    {
      compress_gather(reinterpret_cast<const double *>(dataArray),
                      indices,
                      num_indices,
                      gather_block_size * 2,
                      d_compressed,
                      bits_per_value,
                      stream);
    }

    inline void
    compress_gather(const std::complex<float>   *dataArray,
                    const dftfe::uInt           *indices,
                    size_t                       num_indices,
                    dftfe::uInt                  gather_block_size,
                    void                        *d_compressed,
                    int                          bits_per_value,
                    dftfe::utils::deviceStream_t stream)
    {
      compress_gather(reinterpret_cast<const float *>(dataArray),
                      indices,
                      num_indices,
                      gather_block_size * 2,
                      d_compressed,
                      bits_per_value,
                      stream);
    }

    // Fused decompress+scatter_add complex overloads
    inline void
    decompress_scatter_add(const void                  *d_compressed,
                           const dftfe::uInt           *indices,
                           size_t                       num_indices,
                           dftfe::uInt                  gather_block_size,
                           std::complex<double>        *dataArray,
                           int                          bits_per_value,
                           dftfe::utils::deviceStream_t stream)
    {
      decompress_scatter_add(d_compressed,
                             indices,
                             num_indices,
                             gather_block_size * 2,
                             reinterpret_cast<double *>(dataArray),
                             bits_per_value,
                             stream);
    }

    inline void
    decompress_scatter_add(const void                  *d_compressed,
                           const dftfe::uInt           *indices,
                           size_t                       num_indices,
                           dftfe::uInt                  gather_block_size,
                           std::complex<float>         *dataArray,
                           int                          bits_per_value,
                           dftfe::utils::deviceStream_t stream)
    {
      decompress_scatter_add(d_compressed,
                             indices,
                             num_indices,
                             gather_block_size * 2,
                             reinterpret_cast<float *>(dataArray),
                             bits_per_value,
                             stream);
    }

  } // namespace compressionWrapper
} // namespace dftfe

#endif // DFTFE_WITH_DEVICE
#endif // DFTFE_COMPRESSION_WRAPPER_H
