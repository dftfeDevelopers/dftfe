// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
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
    static const int DEVICE_WARP_SIZE      = 64;
    static const int DEVICE_MAX_BLOCK_SIZE = 1024;
    static const int DEVICE_BLOCK_SIZE     = 512;

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

#    endif

#  endif // dftfeDeviceKernelLauncherConstants_h
#endif   // DFTFE_WITH_DEVICE
