// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------

/*
 * @author Ian C. Lin., Sambit Das
 */
#ifdef DFTFE_WITH_DEVICE
#  ifndef dftfeDeviceKernelLauncherConstants_h
#    define dftfeDeviceKernelLauncherConstants_h

#    ifdef DFTFE_WITH_DEVICE_NVIDIA
namespace dftfe
{
  namespace utils
  {
    static const int DEVICE_WARP_SIZE      = 32;
    static const int DEVICE_MAX_BLOCK_SIZE = 1024;
    static const int DEVICE_BLOCK_SIZE     = 256;

  } // namespace utils
} // namespace dftfe

#    elif DFTFE_WITH_DEVICE_AMD

namespace dftfe
{
  namespace utils
  {
    static const int DEVICE_WARP_SIZE      = 64;
    static const int DEVICE_MAX_BLOCK_SIZE = 1024;
    static const int DEVICE_BLOCK_SIZE     = 512;

  } // namespace utils
} // namespace dftfe

#    elif DFTFE_WITH_DEVICE_INTEL

namespace dftfe
{
  namespace utils
  {
    static const int DEVICE_WARP_SIZE      = 32;
    static const int DEVICE_MAX_BLOCK_SIZE = 1024;
    static const int DEVICE_BLOCK_SIZE     = 256;

  } // namespace utils
} // namespace dftfe

#    endif
#    ifdef DFTFE_WITH_DEVICE_NVIDIA
#      define DFTFE_LAUNCH_KERNEL(kernel, grid, block, shared, stream, ...) \
        do                                                                  \
          {                                                                 \
            kernel<<<grid, block, shared, stream>>>(__VA_ARGS__);           \
        } while (0)
#    elif defined(DFTFE_WITH_DEVICE_AMD)
#      define DFTFE_LAUNCH_KERNEL(kernel, grid, block, shared, stream, ...) \
        do                                                                  \
          {                                                                 \
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel),                     \
                               grid,                                        \
                               block,                                       \
                               shared,                                      \
                               stream,                                      \
                               __VA_ARGS__);                                \
        } while (0)
#    elif defined(DFTFE_WITH_DEVICE_INTEL)
#      define DFTFE_LAUNCH_KERNEL(kernel, grid, block, shared, stream, ...) \
        do                                                                  \
          {                                                                 \
            stream.parallel_for(sycl::nd_range<1>((grid) * (block), block), \
                                [=](sycl::nd_item<1> ind) {                 \
                                  kernel(ind, __VA_ARGS__);                 \
                                });                                         \
        } while (0)
#    else
#      error \
        "No device backend defined (DFTFE_WITH_DEVICE_NVIDIA or DFTFE_WITH_DEVICE_AMD or DFTFE_WITH_DEVICE_INTEL)"
#    endif


#    define DFTFE_KERNEL_NAME(...) __VA_ARGS__


#    if defined(DFTFE_WITH_DEVICE_NVIDIA) || defined(DFTFE_WITH_DEVICE_AMD)
#      define DFTFE_CREATE_KERNEL(RET, NAME, BODY, ...)    \
        __global__ RET NAME(__VA_ARGS__)                   \
        {                                                  \
          const dftfe::uInt globalThreadId =               \
            blockIdx.x * blockDim.x + threadIdx.x;         \
          const dftfe::uInt nThreadsPerBlock = blockDim.x; \
          const dftfe::uInt nThreadBlock     = gridDim.x;  \
          BODY                                             \
        }
#    elif defined(DFTFE_WITH_DEVICE_INTEL)
#      define DFTFE_CREATE_KERNEL(RET, NAME, BODY, ...)                \
        RET NAME(sycl::nd_item<1> ind, __VA_ARGS__)                    \
        {                                                              \
          const dftfe::uInt globalThreadId   = ind.get_global_id(0);   \
          const dftfe::uInt nThreadsPerBlock = ind.get_local_range(0); \
          const dftfe::uInt nThreadBlock     = ind.get_group_range(0); \
          BODY                                                         \
        }
#    else
#      error \
        "No device backend defined (DFTFE_WITH_DEVICE_NVIDIA or DFTFE_WITH_DEVICE_AMD or DFTFE_WITH_DEVICE_INTEL)"
#    endif

#  endif // dftfeDeviceKernelLauncherConstants_h
#endif   // DFTFE_WITH_DEVICE
